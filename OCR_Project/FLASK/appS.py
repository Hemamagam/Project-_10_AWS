from flask import Flask, request, redirect, url_for, render_template, send_file, send_from_directory
import cv2
import numpy as np
import pytesseract as py
import pandas as pd
import os

app = Flask(__name__)

# Set Tesseract executable path if necessary
py.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path as needed

# Ensure the required directories exist
if not os.path.exists('out_csv'):
    os.makedirs('out_csv')
if not os.path.exists('images'):
    os.makedirs('images')

# Load YOLO model
weights_path = r'D:\AI Course Digicrome\One Python\Nexthike-Project Work\Project 10-New-AWS-OCR\OCR_Project\FLASK\model\yolov3-608.weights'
cfg_path = r'D:\AI Course Digicrome\One Python\Nexthike-Project Work\Project 10-New-AWS-OCR\OCR_Project\FLASK\model\yolov3-608.cfg'

# Check if files exist before loading the model
if not os.path.isfile(weights_path):
    print(f"Weights file not found: {weights_path}")
if not os.path.isfile(cfg_path):
    print(f"Config file not found: {cfg_path}")

# Load the model
net = cv2.dnn.readNet(weights_path, cfg_path)

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def process_image(image_path):
    try:
        # Read the image
        image = cv2.imread(image_path)
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        # Load classes
        classes = []
        with open('yolov3-classes.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        # Preprocess the image and pass it through YOLO
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        class_ids, confidences, boxes = [], [], []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # Extract boxes, class IDs, and confidences
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([round(x), round(y), round(w), round(h)])

        # Apply non-max suppression to remove duplicate bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # Prepare the OCR results list
        A = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box
            crop_img = image[y:y + h, x:x + w]

            # Image preprocessing for OCR
            gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            roi = cv2.bitwise_not(thresh)

            # OCR using Tesseract
            c = py.image_to_string(roi, config='--oem 3')
            A.append(c)

        # Process OCR results and save as a CSV
        B = [a.split('\n') for a in A]
        K = ["Test Name", "Unit", "Reference Value", "Value"]
        Data = dict(zip(K, B))
        Y = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in Data.items()]))

        csv_file_path = 'out_csv/extracted_data.csv'
        Y.to_csv(csv_file_path, index=False, header=['Test Name', 'Unit', 'Reference Value', 'Value'])

        return csv_file_path

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        if file:
            # Save the uploaded file
            image_path = os.path.join('images', file.filename)
            file.save(image_path)

            # Store image path temporarily in session or pass to the template
            return render_template('display_image.html', image_path=image_path)

    return '''
    <!doctype html>
    <title>Upload Image</title>
    <h1>Upload an image for OCR</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file required>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/process', methods=['POST'])
def process_and_download():
    image_path = request.form.get('image_path')
    if image_path and os.path.exists(image_path):
        csv_file = process_image(image_path)
        if csv_file:
            # Delete the image after processing (optional)
            os.remove(image_path)
            return send_file(csv_file, as_attachment=True)
        else:
            return "Error processing image. Please try again.", 500
    return "No image found to process.", 400

@app.route('/images/<filename>')
def uploaded_image(filename):
    return send_from_directory('images', filename)

if __name__ == '__main__':
    app.run(debug=True)
