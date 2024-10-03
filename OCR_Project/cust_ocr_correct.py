import cv2
import pytesseract
import numpy as np
from pytesseract import Output
from matplotlib import pyplot as plt
from fuzzywuzzy import fuzz
import pandas as pd  # Import pandas for CSV writing

# Load the image
image_path = 'D:\AI Course Digicrome\One Python\Nexthike-Project Work\Project 10-New-AWS-OCR\OCR_Project\images\thyrocare_0_447.jpg'
image = cv2.imread(image_path)

# Use pytesseract to perform OCR on the image and get box data
custom_config = r'--oem 3 --psm 6'  # Use OCR engine mode 3 and Page Segmentation mode 6
d = pytesseract.image_to_data(image, output_type=Output.DICT, config=custom_config)

# Initialize variables to store bounding box coordinates
detected_boxes = {
    'test_name': [],
    'values': [],
    'units': [],
    'reference_range': []
}

# Define the expected test names, units, and reference ranges
expected_test_names = [
    "total triiodothyronine", "thyroxine", "thyroid stimulating hormone", "triiodothyronine"
]

expected_units = [
    "ng/dl", "ug/dl", "mIU/ml"
]

expected_reference_ranges = [
    "60-200", "4.5-12", "0.3-5.5"
]

# Define a threshold for fuzzy matching
MATCH_THRESHOLD = 80

# Function to perform fuzzy matching on expected values
def fuzzy_match(word, expected_list):
    for expected in expected_list:
        if fuzz.ratio(word.lower(), expected.lower()) > MATCH_THRESHOLD:
            return expected
    return None

# Keep track of the last test name seen to associate with values
last_test_name = None

# Iterate through the detected words to categorize into respective boxes
for i, word in enumerate(d['text']):
    if len(word.strip()) == 0:
        continue

    matched_test_name = fuzzy_match(word, expected_test_names)
    matched_units = fuzzy_match(word, expected_units)
    matched_reference_range = fuzzy_match(word, expected_reference_ranges)

    # Categorize words based on their content and context
    if matched_test_name:
        last_test_name = matched_test_name
        detected_boxes['test_name'].append((d['left'][i], d['top'][i], d['width'][i], d['height'][i], matched_test_name))
    elif matched_reference_range:
        detected_boxes['reference_range'].append((d['left'][i], d['top'][i], d['width'][i], d['height'][i], matched_reference_range))
    elif word.replace('.', '', 1).isdigit():  # Check if the word is a numeric value
        if last_test_name:
            detected_boxes['values'].append((d['left'][i], d['top'][i], d['width'][i], d['height'][i], word, last_test_name))  # Append with the test name
    elif matched_units:
        detected_boxes['units'].append((d['left'][i], d['top'][i], d['width'][i], d['height'][i], matched_units))

# Prepare data for CSV export
csv_data = []

# Group values with their associated test names and units
for value in detected_boxes['values']:
    test_name = value[5]  # The test name associated with the value
    numeric_value = value[4]  # The numeric value
    # Find the corresponding unit for this test name
    unit = None
    for u in detected_boxes['units']:
        if u[1] == value[1]:  # Check if the units' vertical position is similar to the value's position
            unit = u[4]
            break

    # Find the corresponding reference range for this test name
    reference_range = None
    for rr in detected_boxes['reference_range']:
        if rr[1] == value[1]:  # Check if the reference range's vertical position is similar to the value's position
            reference_range = rr[4]
            break

    csv_data.append([test_name, numeric_value, unit, reference_range])

# Create a DataFrame from the collected data
df = pd.DataFrame(csv_data, columns=['Test Name', 'Value', 'Unit', 'Reference Range'])

# Save the DataFrame to a CSV file
csv_output_path = 'D:\AI Course Digicrome\One Python\Nexthike-Project Work\Project 10-New-AWS-OCR\OCR_Project\detected_data.csv'
df.to_csv(csv_output_path, index=False)

# Draw bounding boxes for test names and reference ranges
def get_bounding_box(boxes, padding=2):
    if len(boxes) == 0:
        return None
    x_min = min([box[0] for box in boxes]) - padding
    y_min = min([box[1] for box in boxes]) - padding
    x_max = max([box[0] + box[2] for box in boxes]) + padding
    y_max = max([box[1] + box[3] for box in boxes]) + padding
    return (x_min, y_min, x_max, y_max)

# Draw a single bounding box around each category with padding
def draw_single_box(image, boxes, color, padding=20):
    box = get_bounding_box(boxes, padding)
    if box:
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

# Draw bounding boxes for test names and reference ranges
draw_single_box(image, detected_boxes['test_name'], (0, 255, 0), padding=25)   # Green for Test Name
draw_single_box(image, detected_boxes['reference_range'], (255, 255, 0), padding=10)  # Cyan for Reference Range

# Draw a box for the entire values column
draw_single_box(image, detected_boxes['values'], (255, 0, 0), padding=10)  # Red for Values

# Draw a box for units, ensuring it is associated with values
draw_single_box(image, detected_boxes['units'], (0, 255, 255), padding=10)  # Yellow for Units

# Save the output image with bounding boxes
output_path = 'D:\AI Course Digicrome\One Python\Nexthike-Project Work\Project 10-New-AWS-OCR\OCR_Project\detected_image_with_boxes.jpg'
cv2.imwrite(output_path, image)

print(f"Processed image saved at: {output_path}")
print(f"CSV file saved at: {csv_output_path}")

# Display the final image with bounding boxes using matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Fields with Bounding Boxes")
plt.axis('off')
plt.show()