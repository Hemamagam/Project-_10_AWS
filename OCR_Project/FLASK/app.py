# Import necessary libraries
import os
import cv2
import pytesseract
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from pytesseract import Output
from flask import Flask, request, render_template, redirect, url_for, send_file
import base64

# Initialize Flask app
app = Flask(__name__)

# Define upload folder and results folder
UPLOAD_FOLDER = './uploads'
RESULTS_FOLDER = './results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload and results directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the homepage to upload and preview image
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is part of the POST request
        if 'file' not in request.files:
            return render_template('index.html', error="No file part in the request.")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error="No file selected for uploading.")

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Encode the image to base64 for displaying in HTML
            with open(file_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Render index.html with the uploaded image preview
            return render_template("index.html", uploaded_image=encoded_image, filename=filename, show_preview=True)
    
    # For GET request, simply render the upload form
    return render_template("index.html")

# Route to process the image and display results
@app.route('/process_image/<filename>')
def process_image_route(filename):
    # Construct the file path of the uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # Process the image and get the result paths
    csv_path, processed_image_path = process_image(file_path, filename)

    # Read the processed image for display
    with open(processed_image_path, "rb") as image_file:
        processed_image_encoded = base64.b64encode(image_file.read()).decode('utf-8')

    # Display the result page with the processed image and CSV download link
    return render_template("result.html", processed_image=processed_image_encoded, filename=filename)

# Route to handle CSV download
@app.route('/download_csv/<filename>')
def download_csv(filename):
    csv_path = os.path.join(RESULTS_FOLDER, f"{filename.split('.')[0]}_output.csv")
    
    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True)
    else:
        return "CSV file not found.", 404

# Function to process the image, extract text and draw bounding boxes
def process_image(image_path, filename):
    # Load the image
    image = cv2.imread(image_path)

    # Use pytesseract to perform OCR on the image and get box data
    custom_config = r'--oem 3 --psm 6'
    d = pytesseract.image_to_data(image, output_type=Output.DICT, config=custom_config)

    # Initialize variables to store bounding box coordinates
    detected_boxes = {'test_name': [], 'values': [], 'units': [], 'reference_range': []}

    # Define expected test names, units, and reference ranges
    expected_test_names = ["total triiodothyronine", "thyroxine", "thyroid stimulating hormone", "triiodothyronine"]
    expected_units = ["ng/dl", "ug/dl", "mIU/ml"]
    expected_reference_ranges = ["60-200", "4.5-12", "0.3-5.5"]

    MATCH_THRESHOLD = 80

    # Function to perform fuzzy matching
    def fuzzy_match(word, expected_list):
        for expected in expected_list:
            if fuzz.ratio(word.lower(), expected.lower()) > MATCH_THRESHOLD:
                return expected
        return None

    last_test_name = None

    # Iterate through the detected words
    for i, word in enumerate(d['text']):
        if len(word.strip()) == 0:
            continue

        matched_test_name = fuzzy_match(word, expected_test_names)
        matched_units = fuzzy_match(word, expected_units)
        matched_reference_range = fuzzy_match(word, expected_reference_ranges)

        if matched_test_name:
            last_test_name = matched_test_name
            detected_boxes['test_name'].append((d['left'][i], d['top'][i], d['width'][i], d['height'][i], matched_test_name))
        elif matched_reference_range:
            detected_boxes['reference_range'].append((d['left'][i], d['top'][i], d['width'][i], d['height'][i], matched_reference_range))
        elif word.replace('.', '', 1).isdigit():  # Check if it's a numeric value
            if last_test_name:
                detected_boxes['values'].append((d['left'][i], d['top'][i], d['width'][i], d['height'][i], word, last_test_name))
        elif matched_units:
            detected_boxes['units'].append((d['left'][i], d['top'][i], d['width'][i], d['height'][i], matched_units))

    # Prepare data for CSV export
    csv_data = []
    for value in detected_boxes['values']:
        test_name = value[5]
        numeric_value = value[4]
        unit = next((u[4] for u in detected_boxes['units'] if u[1] == value[1]), None)
        reference_range = next((rr[4] for rr in detected_boxes['reference_range'] if rr[1] == value[1]), None)
        csv_data.append([test_name, numeric_value, unit, reference_range])

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(csv_data, columns=['Test Name', 'Value', 'Unit', 'Reference Range'])
    csv_output_path = os.path.join(RESULTS_FOLDER, f"{filename.split('.')[0]}_output.csv")
    df.to_csv(csv_output_path, index=False)

    # Draw bounding boxes on the image
    def get_bounding_box(boxes, padding=2):
        if len(boxes) == 0:
            return None
        x_min = min([box[0] for box in boxes]) - padding
        y_min = min([box[1] for box in boxes]) - padding
        x_max = max([box[0] + box[2] for box in boxes]) + padding
        y_max = max([box[1] + box[3] for box in boxes]) + padding
        return (x_min, y_min, x_max, y_max)

    def draw_single_box(image, boxes, color, padding=20):
        box = get_bounding_box(boxes, padding)
        if box:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    # Draw boxes for different categories
    draw_single_box(image, detected_boxes['test_name'], (0, 255, 0), padding=25)
    draw_single_box(image, detected_boxes['reference_range'], (255, 255, 0), padding=10)
    draw_single_box(image, detected_boxes['values'], (255, 0, 0), padding=10)
    draw_single_box(image, detected_boxes['units'], (0, 255, 255), padding=10)

    # Save the processed image
    processed_image_path = os.path.join(RESULTS_FOLDER, f"{filename.split('.')[0]}_with_boxes.jpg")
    cv2.imwrite(processed_image_path, image)

    return csv_output_path, processed_image_path

# Start the Flask application
if __name__ == "__main__":
    app.run(debug=True)
