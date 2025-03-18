import cv2
import pytesseract
import numpy as np
from PIL import Image
import os

# Set Tesseract path (Windows users only, update if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path):
    """Preprocess the image to improve OCR accuracy."""

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: The file does not exist at {image_path}")
    
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Error: unable to load image. The file might be corrupted or in an unsupported format: {image_path}")

    # ✅ Debugging Info: Print shape and channels
    print(f"Image loaded successfully. Shape: {image.shape}, Channels: {image.shape[2] if len(image.shape) == 3 else 1}")

    # ✅ If the image has an alpha channel (transparency), remove it
    if len(image.shape) == 3 and image.shape[2] == 4:
        print("Removing alpha channel from image (transparency detected).")
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    #convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ✅ Debugging: Save grayscale image to check output
    cv2.imwrite("debug_grayscale.png", gray)
    print("Grayscale conversion successful.")

    # ✅ Apply thresholding
    try:
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # Fixed threshold
        cv2.imwrite("debug_threshold.png", thresh)
        print("Thresholding successful.")
    except Exception as e:
        raise ValueError(f"Error in thresholding: {e}")

     # ✅ Denoise
    try:
        denoised = cv2.fastNlMeansDenoising(thresh, h=10)  # Lower h-value
        cv2.imwrite("debug_denoised.png", denoised)
        print("Denoising successful.")
    except Exception as e:
        raise ValueError(f"Error in denoising: {e}")

    # ✅ Morphological transformation
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

    # ✅ Check if preprocessing failed
    if processed_image is None:
        raise ValueError("Error: Image preprocessing failed. Check debug images.")

    # ✅ Convert OpenCV (NumPy) image to PIL format before passing to Tesseract
    try:
        pil_image = Image.fromarray(processed_image)
    except Exception as e:
        raise ValueError(f"Error in converting to PIL image: {e}")

    # Extract text
    extracted_text = pytesseract.image_to_string(pil_image)

    return extracted_text.strip()

if __name__ == "__main__":
    input_image = r"C:\Users\bharg\Downloads\invoice.jpg"  # Update your image path

    try:
        extracted_text = extract_text(input_image)
        print("\n✅ Extracted Text:\n", extracted_text)
    except (FileNotFoundError, ValueError) as e:
        print(e)
