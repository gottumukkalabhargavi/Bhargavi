import cv2
import pytesseract
import numpy as np
from PIL import Image
import os

def preprocess_image(image_path):
    """Preprocess the image for better OCR accuracy."""

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: The file does not exist at {image_path}")

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Error: Unable to load the image. Check file format.")

    print(f"Image loaded successfully. Shape: {image.shape}, Channels: {image.shape[2] if len(image.shape) == 3 else 1}")

    if len(image.shape) == 3 and image.shape[2] == 4:
        print("Removing alpha channel from image.")
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    #  Morphological transformation
    try:
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("debug_final.png", morph)
        print("Morphological transformation successful.")
    except Exception as e:
        raise ValueError(f"Error in morphological transformation: {e}")

    return morph

def extract_text(image_path):
    """Extract text from an image using Tesseract OCR."""
    
    try:
        processed_image = preprocess_image(image_path)
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {e}")

    if processed_image is None:
        raise ValueError("Error: Image preprocessing failed. Check debug images.")

    try:
        pil_image = Image.fromarray(processed_image)
    except Exception as e:
        raise ValueError(f"Error in converting to PIL image: {e}")

    extracted_text = pytesseract.image_to_string(pil_image)

    return extracted_text.strip()

if __name__ == "__main__":
    input_image = r"C:\Users\bharg\Downloads\invoice.jpg"

    try:
        extracted_text = extract_text(input_image)
        print("\nExtracted Text:\n", extracted_text)
    except (FileNotFoundError, ValueError) as e:
        print(e)
