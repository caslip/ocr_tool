import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import os
import io
from PyPDF2 import PdfReader
from typing import Optional, Tuple
from skimage.transform import rotate
from skimage import exposure
import datetime
from langchain_core.tools import tool

pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"


def ocr_image(image_path: str, mode: str = "Public", custom_dict_path: Optional[str] = None) -> dict:
    """
    Perform OCR on an image using pytesseract
    
    Parameters:
        mode (str): Public: use tesseract for OCR; Commercial: use commercial OCR service (not implemented)
        image_path (str): Path to the image file
        custom_dict_path (str): Path to custom dictionary file (optional)
        
    Returns:
        dict: Dictionary containing extracted text and average confidence
    """
    try:
        if mode == "Commercial":
            return {"error": "Commercial OCR service not implemented", "text": "", "confidence": 0.0}
        elif mode == "Public":
            model = pytesseract

        # Image preprocessing
        # processed_image = preprocess_image(image_path, custom_dict_path)
        
        # Prepare Tesseract configuration
        config = '--oem 3 --psm 6'
        
        # If custom dictionary is provided, add to configuration
        if custom_dict_path and os.path.exists(custom_dict_path):
            config += f' --user-words "{custom_dict_path}"'
        
        # Perform OCR using pytesseract, get confidence
        lang_param = 'chi_sim+eng'
        
        data = model.image_to_data(image_path, lang=lang_param, config=config, 
                                        output_type=model.Output.DICT)
        
        # Extract text
        text = ""
        confidence_sum = 0
        confidence_count = 0
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # Ignore characters with 0 confidence
                text += data['text'][i]
                if data['text'][i].strip():  # Only calculate confidence for non-whitespace characters
                    confidence_sum += int(data['conf'][i])
                    confidence_count += 1
        
        # Calculate average confidence
        avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0.0
        
        # Save OCR text to raw_data directory
        try:
            # Ensure raw_data directory exists
            raw_data_dir = "raw_data"
            os.makedirs(raw_data_dir, exist_ok=True)
            
            # Generate unique filename using timestamp and original filename
            original_filename = os.path.splitext(os.path.basename(image_path))[0]
            raw_filename = f"{original_filename}_ocr.txt"
            raw_filepath = os.path.join(raw_data_dir, raw_filename)
            
            # Write to file
            with open(raw_filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"OCR text saved to: {raw_filepath}")
        except Exception as e_save:
            print(f"Error saving OCR text: {str(e_save)}")
        
        return {"text": text, "confidence": avg_confidence, "skew_angle": detect_skew(cv2.imread(image_path))}
    except Exception as e:
        return {"error": f"Error during OCR processing: {str(e)}", "text": "", "confidence": 0.0, "skew_angle": 0.0}

def ocr_pdf(pdf_path: str, mode: str = "Public", custom_dict_path: Optional[str] = None) -> dict:
    """
    Perform OCR on each page of a PDF file
    
    Parameters:
        mode (str): Public: use tesseract for OCR; Commercial: use commercial OCR service (not implemented)
        pdf_path (str): Path to the PDF file
        custom_dict_path (str): Path to custom dictionary file (optional)
        
    Returns:
        dict: Dictionary containing extracted text, per-page confidence, and average confidence
    """
    
    doc = ""
    page_confidences = []
    total_confidence_sum = 0
    total_confidence_count = 0

    # Open PDF file
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    # Iterate through each page in the PDF
    for page_number, page in enumerate(reader.pages):
        page_text = ""
        page_confidence = 0.0
        page_confidence_count = 0
        
        # Try to extract text from the page
        text = page.extract_text()
        if text:
            # If text can be extracted directly, add it to the document
            page_text = text
            # For directly extracted text, assume 100% confidence
            page_confidence = 100.0
            page_confidence_count = 1
        else:
            # If the page has no text, try to extract text from images using OCR
            images = page.images
            if images:
                for image_index, img in enumerate(images):
                    # Extract image data from the PDF
                    image = Image.open(io.BytesIO(img.data))
                    
                    # Use OCR to recognize text in the image, get confidence
                    data = pytesseract.image_to_data(image, lang='chi_sim', output_type=pytesseract.Output.DICT)
                    
                    page_text_part = ""
                    confidence_sum = 0
                    confidence_count = 0
                    
                    for i in range(len(data['text'])):
                        if int(data['conf'][i]) > 0:  # Ignore characters with 0 confidence
                            page_text_part += data['text'][i]
                            if data['text'][i].strip():  # Only calculate confidence for non-whitespace characters
                                confidence_sum += int(data['conf'][i])
                                confidence_count += 1
                    
                    # Calculate average confidence for the current image
                    image_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0.0
                    
                    # Update page text and confidence
                    page_text += (f"Page {page_number+1}, Image {image_index+1}: {page_text_part}")
                    
                    # Accumulate to page confidence
                    if confidence_count > 0:
                        total_confidence_sum += confidence_sum
                        total_confidence_count += confidence_count
                        page_confidence_count += confidence_count
                        page_confidence = total_confidence_sum / total_confidence_count if total_confidence_count > 0 else 0.0
            else:
                # If the page has no text or images, add a placeholder
                page_text = (f"Page {page_number+1} has no text or images.")
                page_confidence = 0.0
                page_confidence_count = 1
        
        # Add page text to the document
        doc += page_text + "\n"
        
        # Record page confidence
        page_confidences.append({
            "page_number": page_number + 1,
            "confidence": page_confidence,
            "text_length": len(page_text)
        })
    
    # Calculate overall average confidence
    overall_confidence = total_confidence_sum / total_confidence_count if total_confidence_count > 0 else 0.0
    
    # Save OCR text to raw_data directory
    try:
        # Ensure raw_data directory exists
        raw_data_dir = "raw_data"
        os.makedirs(raw_data_dir, exist_ok=True)
        
        original_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        raw_filename = f"{original_filename}_ocr.txt"
        raw_filepath = os.path.join(raw_data_dir, raw_filename)
        
        # Write to file
        with open(raw_filepath, 'w', encoding='utf-8') as f:
            f.write(doc)
        
        print(f"OCR text saved to: {raw_filepath}")
    except Exception as e_save:
        print(f"Error saving OCR text: {str(e_save)}")
    
    # Return results
    result = {
        "text": doc,
        "total_pages": total_pages,
        "page_confidences": page_confidences,
        "overall_confidence": overall_confidence,
        "skew_angle": 0.0  # PDF processing does not calculate skew angle
    }
    return result

def detect_skew(image: np.ndarray) -> float:
    """
    Detect the skew angle of an image
    
    Parameters:
        image (np.ndarray): Input image
        
    Returns:
        float: Detected skew angle in degrees
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
    
    # Calculate median angle
    if angles:
        median_angle = float(np.median(angles))
        # Only consider angles between -30 and 30 degrees
        if -30 <= median_angle <= 30:
            return median_angle
    
    return 0.0
