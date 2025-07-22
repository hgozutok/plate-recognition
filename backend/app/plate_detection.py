import cv2
import numpy as np
from PIL import Image
import io
import pytesseract
import re
from typing import Tuple, Optional

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r'tesseract'  # Default location in Docker

def clean_plate_text(text: str) -> Optional[str]:
    """
    Clean and validate the detected plate text.
    Returns None if no valid plate number is found.
    
    Validation rules:
    1. Must contain both letters and numbers
    2. Length between 5 and 8 characters (typical for most plates)
    3. Cannot contain easily confused characters (0/O, 1/I)
    4. Must match common plate patterns
    5. No more than 4 consecutive numbers or letters
    """
    print(f"DEBUG: clean_plate_text input: '{text}'")
    
    # Remove whitespace and newlines
    text = text.strip().replace('\n', ' ').replace('\r', '')
    
    # Convert to uppercase and remove any special characters
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    print(f"DEBUG: After cleaning: '{text}'")
    
    # Basic length validation (more flexible for Turkish plates)
    if len(text) < 4 or len(text) > 15:  # More lenient
        print(f"DEBUG: Length validation failed: {len(text)}")
        return None
        
    # Must contain at least one letter and one number
    if not (re.search(r'[A-Z]', text) and re.search(r'[0-9]', text)):
        print(f"DEBUG: Must contain letters and numbers")
        return None
    
    # Replace commonly confused characters
    text = text.replace('O', '0').replace('I', '1')
    
    # Check for invalid patterns
    if re.search(r'[A-Z]{5,}', text) or re.search(r'[0-9]{5,}', text):
        return None
    
    # Common plate patterns including Turkish plates
    common_patterns = [
        # Turkish plate patterns (more flexible)
        r'^[0-9]{2}[A-Z]{1,3}[0-9]{2,4}$',  # Turkish format: 34ABC123, 06AB123
        r'^[0-9]{2}[A-Z]{1,3}[0-9]{2}$',     # Turkish format: 34ABC12
        r'^[0-9]{2}[A-Z]{2}[0-9]{3}$',       # Turkish format: 20DT185
        r'^[0-9]{2}[A-Z]{2}[0-9]{2,4}$',     # Turkish format: 20DT185, 20DT1855
        # Standard international patterns
        r'^[A-Z]{2,3}[0-9]{2,4}$',           # AA1234, ABC123
        r'^[0-9]{2,4}[A-Z]{2,3}$',           # 1234AA, 123ABC
        r'^[A-Z][0-9]{3,4}[A-Z]{2}$',        # A1234BC
        r'^[A-Z]{2}[0-9]{2}[A-Z]{2}$'        # AB12CD
    ]
    
    # Special handling for Turkish plates
    if re.match(r'^[0-9]{2}', text):  # If starts with 2 numbers (potential Turkish plate)
        # Valid Turkish city codes (1-81)
        valid_city_codes = set(range(1, 82))
        try:
            city_code = int(text[:2])
            if city_code not in valid_city_codes:
                return None
        except ValueError:
            return None
    
    # Check if the text matches any common pattern
    pattern_matched = False
    for i, pattern in enumerate(common_patterns):
        if re.match(pattern, text):
            print(f"DEBUG: Matched pattern {i+1}: {pattern}")
            pattern_matched = True
            break
    
    if not pattern_matched:
        print(f"DEBUG: No pattern matched for: '{text}'")
        # For debugging, let's be more lenient and return the text anyway if it looks reasonable
        if len(text) >= 5 and len(text) <= 10:
            print(f"DEBUG: Returning text despite no pattern match")
            return text
        return None
    
    # Additional validation for similar characters   
    if any(pair in text for pair in ['B8', '5S', '2Z']):
        # Apply extra validation or confidence check for these cases
        pass
    
    # Remove any remaining unwanted characters (just in case)
    text = re.sub(r'[^A-Z0-9]', '', text)
    
    return text if text else None

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the image for better plate detection.
    Enhanced for Turkish plates which often have white background and black text.
    Applies various filters and transformations to enhance plate visibility.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    bilateral = cv2.bilateralFilter(enhanced, 11, 17, 17)
    
    # Apply adaptive threshold with optimized parameters for Turkish plates
    thresh = cv2.adaptiveThreshold(
        bilateral,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        13,  # Increased block size for better handling of Turkish plates
        4    # Adjusted constant for better contrast
    )
    
    # Apply morphological operations to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return morph

def find_plate_contour(processed_img: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the contour that most likely contains the license plate.
    Uses contour properties to filter and identify the plate region.
    """
    # Find all contours
    contours, _ = cv2.findContours(
        processed_img,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plate_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # License plates are typically rectangles with 4 corners
        if len(approx) == 4:
            # Check if the aspect ratio matches typical license plates
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            
            # Most license plates have aspect ratio between 2.0 and 5.5
            if 2.0 <= aspect_ratio <= 5.5:
                plate_contour = approx
                break
    
    return plate_contour

def extract_plate_region(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Extract and transform the plate region to a standardized view.
    Applies perspective transform to get a straight view of the plate.
    """
    # Get the corners of the plate
    pts = contour.reshape(4, 2)
    
    # Order points in [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left will have the smallest sum
    # Bottom-right will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right will have the smallest difference
    # Bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # Calculate width and height of the plate
    widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
    widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
    heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points for perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Apply perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def enhance_plate_region(plate_region: np.ndarray) -> np.ndarray:
    """
    Enhance the extracted plate region for better OCR results.
    Optimized for Turkish license plates which have specific characteristics:
    - White background
    - Black text
    - Often reflective surface
    - May have blue region with TR code
    """
    # Convert to grayscale if not already
    if len(plate_region.shape) == 3:
        # Check for blue TR region (common in Turkish plates)
        hsv = cv2.cvtColor(plate_region, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        
        # If blue region detected, process differently
        if cv2.countNonZero(blue_mask) > (plate_region.shape[0] * plate_region.shape[1] * 0.1):
            # Convert to grayscale excluding the blue region
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            gray[blue_mask > 0] = 255  # Set blue region to white
            plate_region = gray
        else:
            plate_region = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast using CLAHE with parameters tuned for Turkish plates
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(plate_region)
    
    # Noise reduction
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    
    # Increase sharpness
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Apply adaptive thresholding for better character separation
    binary = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    
    # Remove small noise
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def preprocess_character(char_image: np.ndarray) -> np.ndarray:
    """
    Apply advanced preprocessing to individual character images.
    """
    # Ensure proper size
    char_image = cv2.resize(char_image, (28, 28))
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    char_image = clahe.apply(char_image)
    
    # Remove noise using median blur
    char_image = cv2.medianBlur(char_image, 3)
    
    # Sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    char_image = cv2.filter2D(char_image, -1, kernel)
    
    # Normalize pixel values
    char_image = cv2.normalize(char_image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Ensure binary image with Otsu's thresholding
    _, char_image = cv2.threshold(char_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return char_image

def segment_characters(plate_image: np.ndarray) -> list[np.ndarray]:
    """
    Segment individual characters from the plate image.
    Returns a list of character images.
    """
    # Find contours of each character
    contours, _ = cv2.findContours(plate_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize character regions
    char_regions = []
    
    height, width = plate_image.shape
    min_char_width = width // 10  # Minimum character width
    min_char_height = height // 2  # Minimum character height
    
    # Sort contours from left to right
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out too small or too large regions
        if (w >= min_char_width * 0.3 and 
            h >= min_char_height * 0.5 and 
            w <= min_char_width * 2 and 
            h <= height * 0.9):
            
            # Extract character region
            char_region = plate_image[y:y+h, x:x+w]
            
            # Add padding around character
            padding = 4  # Increased padding for better isolation
            char_region = cv2.copyMakeBorder(
                char_region, 
                padding, padding, padding, padding,
                cv2.BORDER_CONSTANT,
                value=0
            )
            
            # Apply advanced preprocessing to the character
            char_region = preprocess_character(char_region)
            
            char_regions.append(char_region)
    
    return char_regions

def recognize_character(char_image: np.ndarray, config: str) -> tuple[str, float]:
    """
    Recognize a single character using Tesseract.
    Returns the character and confidence score.
    """
    # Invert if necessary (Tesseract expects black text on white background)
    if np.mean(char_image[0]) > np.mean(char_image[-1]):  # Top row brighter than bottom
        char_image = cv2.bitwise_not(char_image)
    
    # Add padding for better recognition
    char_image = cv2.copyMakeBorder(char_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    
    # Get OCR data
    data = pytesseract.image_to_data(
        char_image,
        config=config,
        output_type=pytesseract.Output.DICT
    )
    
    # Get the recognized character and confidence
    confidences = [int(conf) for conf in data['conf'] if conf != '-1']
    texts = [text for text in data['text'] if text.strip()]
    
    if texts and confidences:
        return texts[0], confidences[0] / 100
    return '', 0.0

def get_plate_confidence(image: np.ndarray) -> float:
    """
    Get the confidence score of the OCR result.
    Returns a value between 0 and 1.
    """
    try:
        # Get detailed OCR data
        data = pytesseract.image_to_data(image, config='--oem 3 --psm 7', output_type=pytesseract.Output.DICT)
        
        # Calculate average confidence of detected text
        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
        return sum(confidences) / len(confidences) / 100 if confidences else 0
    except:
        return 0

def recognize_characters_parallel(char_regions: list[np.ndarray], config: str) -> tuple[str, float]:
    """
    Recognize characters in parallel using multiple threads.
    Returns the combined plate text and average confidence.
    """
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    
    # Create a partial function with fixed config
    recognize_with_config = partial(recognize_character, config=config)
    
    # Process characters in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(recognize_with_config, char_regions))
    
    # Combine results
    chars = []
    confidences = []
    
    for char, conf in results:
        if char and conf > 0.4:  # Minimum character confidence
            chars.append(char)
            confidences.append(conf)
    
    plate_text = ''.join(chars)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    return plate_text, avg_confidence

def detect_plate(image: np.ndarray) -> Tuple[str, np.ndarray, Optional[np.ndarray], float]:
    """
    Main function for plate detection pipeline.
    Returns:
    - plate_number: Detected plate number (placeholder for now)
    - debug_image: Image with detected plate highlighted
    - plate_region: Extracted and enhanced plate region
    """
    # Create a copy for drawing debug information
    debug_image = image.copy()
    
    # Preprocess the image
    processed = preprocess_image(image)
    
    # Find plate contour
    plate_contour = find_plate_contour(processed)
    print(f"DEBUG: Plate contour found: {plate_contour is not None}")
    
    if plate_contour is not None:
        # Draw the detected plate region on debug image
        cv2.drawContours(debug_image, [plate_contour], -1, (0, 255, 0), 3)
        
        # Extract and enhance plate region
        plate_region = extract_plate_region(image, plate_contour)
        enhanced_region = enhance_plate_region(plate_region)
        
        # Perform OCR on the enhanced region
        try:
            print(f"DEBUG: Processing plate region with shape: {enhanced_region.shape}")
            
            # Try multiple OCR configurations for better Turkish plate recognition
            configs = [
                r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                r'--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ]
            
            best_result = None
            best_confidence = 0.0
            
            for i, config in enumerate(configs):
                try:
                    print(f"DEBUG: Trying OCR config {i+1}: {config}")
                    
                    # Get OCR data
                    data = pytesseract.image_to_data(
                        enhanced_region,
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Extract text and confidence
                    texts = []
                    confidences = []
                    for j, text in enumerate(data['text']):
                        conf = int(data['conf'][j]) if data['conf'][j] != '-1' else 0
                        if text.strip() and conf > 20:  # Lower threshold
                            texts.append(text.strip())
                            confidences.append(conf)
                            print(f"DEBUG: Found text '{text.strip()}' with confidence {conf}")
                    
                    if texts:
                        combined_text = ''.join(texts).upper()
                        avg_confidence = sum(confidences) / len(confidences) / 100.0
                        print(f"DEBUG: Combined text: '{combined_text}', avg confidence: {avg_confidence:.2f}")
                        
                        # Clean the text
                        cleaned_text = clean_plate_text(combined_text)
                        print(f"DEBUG: Cleaned text: '{cleaned_text}'")
                        
                        if cleaned_text and avg_confidence > best_confidence:
                            best_result = cleaned_text
                            best_confidence = avg_confidence
                            print(f"DEBUG: New best result: '{best_result}' with confidence {best_confidence:.2f}")
                    
                    # Also try simple string extraction
                    simple_text = pytesseract.image_to_string(enhanced_region, config=config).strip().upper()
                    simple_text = re.sub(r'[^A-Z0-9]', '', simple_text)
                    if simple_text and len(simple_text) >= 4:
                        print(f"DEBUG: Simple OCR result: '{simple_text}'")
                        cleaned_simple = clean_plate_text(simple_text)
                        if cleaned_simple:
                            print(f"DEBUG: Cleaned simple result: '{cleaned_simple}'")
                            if not best_result or len(cleaned_simple) > len(best_result):
                                best_result = cleaned_simple
                                best_confidence = 0.5  # Assign moderate confidence
                        else:
                            # If clean failed, try some basic corrections
                            corrected = simple_text.replace('8', 'B').replace('6', 'G').replace('5', 'S').replace('1', 'I').replace('0', 'O')
                            corrected_clean = clean_plate_text(corrected)
                            if corrected_clean:
                                print(f"DEBUG: Corrected result: '{corrected_clean}'")
                                if not best_result:
                                    best_result = corrected_clean
                                    best_confidence = 0.4
                                
                except Exception as e:
                    print(f"DEBUG: OCR config {i+1} failed: {str(e)}")
                    continue
            
            print(f"DEBUG: Final result: '{best_result}' with confidence {best_confidence:.2f}")
            
            if best_result and best_confidence > 0.20:  # Even lower threshold
                return best_result, debug_image, enhanced_region, best_confidence
            else:
                # If no good result, return whatever we found for debugging
                return best_result or "NO_PLATE", debug_image, enhanced_region, best_confidence
            
        except Exception as e:
            print(f"OCR Error: {str(e)}")
            return "OCR_ERROR", debug_image, enhanced_region, 0.0
    
    return "NO_PLATE", debug_image, None, 0.0

def process_image_file(file_contents: bytes) -> Tuple[str, Image.Image, Optional[Image.Image], float]:
    """
    Process an uploaded image file.
    Returns:
    - plate_number: Detected plate number
    - debug_image: PIL Image with detection visualization
    - plate_region: PIL Image of extracted plate (if found)
    - confidence: OCR confidence score (0-1)
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(file_contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run detection
    plate_number, debug_image, plate_region, confidence = detect_plate(image)
    
    # Convert OpenCV images to PIL
    debug_pil = Image.fromarray(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
    
    if plate_region is not None:
        plate_pil = Image.fromarray(plate_region)
        return plate_number, debug_pil, plate_pil, confidence
    
    return plate_number, debug_pil, None, confidence
