# Project _10_AWS
 Project _10_AWS

# Custom Object Character Recognition (OCR) with Flask using YOLOv3 and Tesseract

# Combining YOLOv3 and Tesseract for Lab Report Analysis

# Project Overview

Objective: To build a custom OCR solution that extracts specific content from lab reports, such as test names, values, units, and reference ranges, converting them into editable files.
Goal: Convert lab report data into editable formats.
Key Technologies: YOLOv3, Tesseract, AWS

# Model and Dataset

Model: Custom-trained YOLOv3 for object detection.
Dataset: Lab report images with labeled regions for OCR.
Dataset Link: Drive Link

# Key Workflow Breakdown

Object Detection: YOLOv3 identifies regions of interest in the lab report.
Image Preprocessing: Resize, grayscale, and apply filters for OCR accuracy.
Text Extraction: Tesseract processes preprocessed images to extract text.
Result Storage: Output saved as CSV

# Flask Integration in Custom OCR Solution
Purpose: Flask is used to create a web-based interface for the OCR project, allowing users to upload images, process them, and visualize results interactively.

# Key Features:

Image Upload & Preview: Users can upload images through a form on the homepage. Flask saves the uploaded files and provides a preview of the image.
Image Processing: The uploaded image is processed using the YOLOv3 and Tesseract combination. Test names, values, units, and reference ranges are extracted, and bounding boxes are drawn on the image.
CSV Output: Extracted data is stored in a CSV file for download. The file contains detailed information, including test names, values, units, and reference ranges.
Visualization: Flask renders the processed image with bounding boxes and provides a download link for the CSV file.




