import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance
import re
from typing import Tuple, Optional, List, Dict
import json

class RobloxDialogOCR:
	def __init__(self, config_path: Optional[str] = None):
		"""
		Initialize the Roblox dialog OCR system
		
		Args:
			config_path: Path to JSON config file with dialog box coordinates
		"""
		# Default dialog box region (you'll need to adjust these)
		self.default_dialog_region = {
			'x': 1150,      # Adjust based on your Roblox window size
			'y': 280,      # Typical dialog position
			'height': 80,  # Height of dialog
			'width': 250  # Width of dialog
		}
		
		# Load custom region if config provided
		print("st - 0.01")
		if config_path:
			self.load_config(config_path)
		else:
			self.dialog_region = self.default_dialog_region
			
		# Tesseract configuration optimized for game text
		self.tesseract_config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;"\'-() '
		
		# Common Roblox UI colors (you may need to adjust these)
		self.text_colors = {
			'white': (255, 255, 255),
			'black': (0, 0, 0),
			'dark_gray': (50, 50, 50),
			'light_gray': (200, 200, 200)
		}
		
	def load_config(self, config_path: str):
		"""Load dialog region configuration from JSON file"""
		try:
			with open(config_path, 'r') as f:
				config = json.load(f)
				self.dialog_region = config.get('dialog_region', self.default_dialog_region)
		except (FileNotFoundError, json.JSONDecodeError):
			print(f"Could not load config from {config_path}, using defaults")
			self.dialog_region = self.default_dialog_region
	
	def extract_dialog_region(self, image: np.ndarray) -> np.ndarray:
		"""
		Extract the dialog box region from the screenshot
		
		Args:
			image: Full screenshot as numpy array
			
		Returns:
			Cropped dialog region
		"""
		print("st - 0.2")
		height, width = image.shape[:2]
		
		print("st - 0.3")
		# Ensure coordinates are within image bounds
		x = max(0, min(self.dialog_region['x'], width - 1))
		y = max(0, min(self.dialog_region['y'], height - 1))
		w = min(self.dialog_region['width'], width - x)
		h = min(self.dialog_region['height'], height - y)
		
		# Extract region
		dialog_crop = image[y:y+h, x:x+w]
		# # Convert from BGR to RGB
		# dialog_crop_rgb = cv2.cvtColor(dialog_crop, cv2.COLOR_BGR2RGB)
		# # Convert to PIL Image
		# pil_image = Image.fromarray(dialog_crop_rgb)
		# # Show the image
		# pil_image.show()


		print("st - 0.4")
		return dialog_crop
	
	def auto_detect_dialog_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
		"""
		Automatically detect dialog box region using contour detection
		This is useful if dialog position varies
		
		Returns:
			(x, y, width, height) or None if not found
		"""
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
		
		# Apply Gaussian blur
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		
		# Edge detection
		edges = cv2.Canny(blurred, 50, 150)
		
		# Find contours
		contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		# Look for rectangular contours that could be dialog boxes
		for contour in sorted(contours, key=cv2.contourArea, reverse=True):
			# Approximate contour
			epsilon = 0.02 * cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, epsilon, True)
			
			# Check if it's roughly rectangular and large enough
			if len(approx) >= 4:
				x, y, w, h = cv2.boundingRect(contour)
				area = w * h
				
				# Filter by size (adjust these thresholds based on your screenshots)
				if area > 50000 and w > h and w > 300 and h > 100:
					return (x, y, w, h)
		
		return None
	
	def preprocess_dialog_image(self, dialog_region: np.ndarray) -> np.ndarray:
		"""
		Preprocess the dialog region for optimal OCR
		
		Args:
			dialog_region: Cropped dialog box image
			
		Returns:
			Preprocessed image
		"""
		# Convert to grayscale if needed
		if len(dialog_region.shape) == 3:
			gray = cv2.cvtColor(dialog_region, cv2.COLOR_BGR2GRAY)
		else:
			gray = dialog_region.copy()
		
		# Enhance contrast
		enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
		
		# Apply Gaussian blur to smooth text
		blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
		
		# Adaptive thresholding for better text extraction
		# Try different threshold methods
		thresh1 = cv2.adaptiveThreshold(
			blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
			cv2.THRESH_BINARY, 11, 2
		)
		
		thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		
		# Choose the better threshold (you might need to tune this)
		# For now, we'll use OTSU as it often works well with game text
		return thresh2
	
	def enhance_text_contrast(self, image: np.ndarray) -> np.ndarray:
		"""
		Enhance text contrast specifically for Roblox dialog styling
		"""
		# Convert to PIL for easier manipulation
		pil_image = Image.fromarray(image)
		
		# Enhance contrast
		enhancer = ImageEnhance.Contrast(pil_image)
		enhanced = enhancer.enhance(1.5)
		
		# Enhance sharpness
		sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
		sharpened = sharpness_enhancer.enhance(1.2)
		
		# Convert back to numpy
		return np.array(sharpened)
	
	def clean_extracted_text(self, text: str) -> str:
		"""
		Clean up OCR output for Roblox dialog text
		
		Args:
			text: Raw OCR output
			
		Returns:
			Cleaned text
		"""
		# Remove extra whitespace
		cleaned = re.sub(r'\s+', ' ', text.strip())
		
		# Common OCR corrections for game text
		corrections = {
			'|': 'I',
			'0': 'O',  # Sometimes O is read as 0
			'1': 'l',  # Sometimes l is read as 1
			'5': 'S',  # Sometimes S is read as 5
		}
		
		# Apply corrections cautiously (only if it makes sense in context)
		for wrong, right in corrections.items():
			# Only replace if it's at word boundaries to avoid over-correction
			cleaned = re.sub(rf'\b{re.escape(wrong)}\b', right, cleaned)
		
		# Remove common OCR artifacts
		cleaned = re.sub(r'[^\w\s.,!?:;"\'-()]', '', cleaned)
		
		return cleaned.strip()
	
	def extract_text_multiple_methods(self, processed_image: np.ndarray) -> List[str]:
		"""
		Extract text using multiple OCR configurations for better accuracy
		"""
		results = []
		
		# Convert to PIL Image
		pil_image = Image.fromarray(processed_image)
		
		# Method 1: Standard configuration
		try:
			text1 = pytesseract.image_to_string(pil_image, config=self.tesseract_config)
			if text1.strip():
				results.append(self.clean_extracted_text(text1))
		except:
			pass
		
		# Method 2: Different PSM mode
		try:
			config2 = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;"\'-() '
			text2 = pytesseract.image_to_string(pil_image, config=config2)
			if text2.strip():
				results.append(self.clean_extracted_text(text2))
		except:
			pass
		
		# Method 3: Line-by-line extraction
		try:
			config3 = '--psm 13'
			text3 = pytesseract.image_to_string(pil_image, config=config3)
			if text3.strip():
				results.append(self.clean_extracted_text(text3))
		except:
			pass
		
		return results
	
	def select_best_result(self, results: List[str]) -> str:
		"""
		Select the best OCR result from multiple attempts
		"""
		if not results:
			return ""
		
		# Filter out very short results (likely errors)
		valid_results = [r for r in results if len(r.strip()) > 2]
		
		if not valid_results:
			return results[0] if results else ""
		
		# Return the longest reasonable result
		# (assumes longer results are more complete)
		best_result = max(valid_results, key=len)
		
		return best_result
	
	def read_dialog_text(self, screenshot_path: str, auto_detect: bool = False) -> Dict:
		"""
		Main method to extract text from Roblox dialog screenshot
		
		Args:
			screenshot_path: Path to the screenshot image
			auto_detect: Whether to auto-detect dialog region
			
		Returns:
			Dictionary with extracted text and metadata
		"""
		try:
			# Load image
			image = cv2.imread(screenshot_path)
			return self.read_dialog_text_from_array(image)
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e),
				'text': '',
				'confidence': 0
			}
	
	def read_dialog_text_from_array(self, image: np.ndarray, auto_detect: bool = False) -> Dict:
		"""
		Main method to extract text from a given screenshot image array (not a file path)
		
		Args:
			screenshot_path: Path to the screenshot image
			auto_detect: Whether to auto-detect dialog region
			
		Returns:
			Dictionary with extracted text and metadata
		"""
		try:
			if image is None:
				return {
					'success': False,
					'error': 'Could not load image',
					'text': '',
					'confidence': 0
				}
			print("st - 0.1")

			# Extract dialog region
			if auto_detect:
				detected_region = self.auto_detect_dialog_region(image)
				if detected_region:
					x, y, w, h = detected_region
					dialog_region = image[y:y+h, x:x+w]
				else:
					# Fall back to default region
					dialog_region = self.extract_dialog_region(image)
			else:
				dialog_region = self.extract_dialog_region(image)
			
			print("st-2")
			
			# Preprocess the dialog region
			processed_image = self.preprocess_dialog_image(dialog_region)
			
			# Enhance contrast
			enhanced_image = self.enhance_text_contrast(processed_image)
			
			# Extract text using multiple methods
			ocr_results = self.extract_text_multiple_methods(enhanced_image)
			
			# Select best result
			final_text = self.select_best_result(ocr_results)
			
			return {
				'success': True,
				'text': final_text,
				'all_results': ocr_results,
				'dialog_region_shape': dialog_region.shape,
				'auto_detected': auto_detect and 'detected_region' in locals()
			}
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e),
				'text': '',
				'confidence': 0
			}

	def calibrate_dialog_region(self, screenshot_path: str, show_preview: bool = True) -> Dict:
		"""
		Helper method to calibrate dialog region coordinates
		Shows the extracted region for manual adjustment
		"""
		image = cv2.imread(screenshot_path)
		if image is None:
			return {'error': 'Could not load image'}
		
		# Extract current region
		dialog_region = self.extract_dialog_region(image)
		
		if show_preview:
			# Show the extracted region
			cv2.imshow('Dialog Region', dialog_region)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		
		return {
			'region_coordinates': self.dialog_region,
			'extracted_shape': dialog_region.shape,
			'success': True
		}
	
	def save_config(self, config_path: str):
		"""Save current dialog region configuration to JSON file"""
		config = {
			'dialog_region': self.dialog_region
		}
		
		with open(config_path, 'w') as f:
			json.dump(config, f, indent=2)

def main():
	ocr = RobloxDialogOCR('dialog_config_4.json')
	screenshot_path = 'roblox_screenshot_3.png'
	print("st 1")
	result = ocr.read_dialog_text(screenshot_path)
	
	if result['success']:
		print(f"Extracted text: '{result['text']}'")
		print(f"All OCR attempts: {result['all_results']}")
	else:
		print(f"Error: {result['error']}")
	
	# Calibration helper (uncomment to use)
	# ocr.calibrate_dialog_region(screenshot_path)

def is_this_phase_2_or_3(image: np.ndarray) -> bool:
    ocr = RobloxDialogOCR('dialog_config_2.json')
    result = ocr.read_dialog_text_from_array(image)
    if result['success']:
        if result['text'] == 'With...':
            return 2
        elif result['text'] == 'And a...':
            return 3
    return 0

def is_this_phase_4(image: np.ndarray) -> bool:
    ocr = RobloxDialogOCR('dialog_config_4.json')
    result = ocr.read_dialog_text_from_array(image)
    if result['success']:
        return result['text'] == 'Canyourepeat?'
    return False

def get_current_phase(image: np.ndarray) -> int:
	if is_this_phase_4(image):
		return 4
	phase_2_or_3 = is_this_phase_2_or_3(image)
	if (phase_2_or_3):
		return phase_2_or_3
	return 1


if __name__ == "__main__":
	# Run main function
	main()
