# Project _10_AWS
 Project _10_AWS

### Custom Object Character Recognition (OCR) with Flask using YOLOv3 and Tesseract

### Combining YOLOv3 and Tesseract for Lab Report Analysis

### Project Overview

##### Objective: To build a custom OCR solution that extracts specific content from lab reports, such as test names, values, units, and reference ranges, converting them into editable files.
##### Goal: Convert lab report data into editable formats.
##### Key Technologies: YOLOv3, Tesseract,OpenCV, Fuzzywuzzy, AWS

### Model and Dataset

##### Model: Custom-trained YOLOv3 for object detection.
##### Dataset: Lab report images with labeled regions for OCR.
##### Dataset Link: https://docs.google.com/document/d/1faS1owAT8EyTTMGtNSeiOLTH3HWEK_0kRSImuxKP2K0/edit 

### Key Workflow Breakdown

##### Object Detection: YOLOv3 identifies regions of interest in the lab report.
##### Image Preprocessing: Resize, grayscale, and apply filters for OCR accuracy.
##### Text Extraction: Tesseract processes preprocessed images to extract text.
##### Result Storage: Output saved as CSV

### Technologies Used:
##### YOLOv3: Object detection for identifying regions of interest in lab reports.
##### Tesseract: OCR engine for extracting text from detected regions.
##### OpenCV: Image preprocessing and manipulation.
##### Fuzzy Matching: Used to enhance detection accuracy for test names, units, and reference ranges.

### Python Libraries:
##### pytesseract for OCR
##### fuzzywuzzy for fuzzy text matching
##### pandas for data export to CSV

### Key Steps:
### Image Processing:

##### Loaded the lab report image using OpenCV.
##### Applied bounding box detection using YOLOv3 to identify regions of interest (test names, values, etc.).
##### Text Detection and Matching:

##### Used Tesseract to extract text from the detected regions.
##### Performed fuzzy matching to compare the extracted text with predefined expected test names, units, and reference ranges.

### Categorization:

##### Categorized the detected content into different categories:
##### Test names
##### Values
##### Units
##### Reference ranges
##### Data Output:

##### The extracted information was saved in a CSV file containing the test name, value, unit, and reference range for each entry.
##### The processed image was saved with bounding boxes drawn around detected fields for visualization.
##### Visualization:

##### Used Matplotlib to display the final image with bounding boxes for better interpretation of the detected regions.

##### Results:
##### Successfully extracted and categorized content from lab reports.
##### Exported results as structured CSV files.
##### Visualized the detected regions in the image for quality verification.

### Flask Integration in Custom OCR Solution
##### Purpose: Flask is used to create a web-based interface for the OCR project, allowing users to upload images, process them, and visualize results interactively.

### Key Features:

##### Image Upload & Preview: Users can upload images through a form on the homepage. Flask saves the uploaded files and provides a preview of the image.
##### Image Processing: The uploaded image is processed using the YOLOv3 and Tesseract combination. Test names, values, units, and reference ranges are extracted, and bounding boxes are drawn on the image.
##### CSV Output: Extracted data is stored in a CSV file for download. The file contains detailed information, including test names, values, units, and reference ranges.
##### Visualization: Flask renders the processed image with bounding boxes and provides a download link for the CSV file.




