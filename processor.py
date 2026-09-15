import cv2
import numpy as np
import pytesseract
from PIL import Image

class ForensicProcessor:
    """
    A class to handle forensic-style image enhancement for indented writing recovery.
    """
    
    @staticmethod
    def to_gray(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def apply_clahe(self, img, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Applies Contrast Limited Adaptive Histogram Equalization.
        This is crucial for bringing out subtle textures in paper.
        """
        gray = self.to_gray(img)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray)

    def virtual_lighting(self, img, angle_deg=45):
        """
        Simulates oblique lighting from a specific angle.
        Uses Sobel gradients to detect surface variations (valleys of pen strokes).
        """
        gray = self.to_gray(img).astype(np.float32)
        
        # Convert angle to radians
        angle_rad = np.deg2rad(angle_deg)
        
        # Calculate gradients in X and Y directions
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Simulate light direction vector
        lx = np.cos(angle_rad)
        ly = np.sin(angle_rad)
        
        # Dot product of gradient and light vector
        # This highlights edges facing the virtual 'light'
        shaded = gx * lx + gy * ly
        
        # Normalize back to 0-255
        shaded = cv2.normalize(shaded, None, 0, 255, cv2.NORM_MINMAX)
        return shaded.astype(np.uint8)

    def denoise(self, img, strength=10):
        """
        Applies a Bilateral Filter to remove paper grain noise while
        preserving the sharp edges of text indentations.
        """
        return cv2.bilateralFilter(img, d=9, sigmaColor=strength, sigmaSpace=strength)

    def apply_negative(self, img):
        """
        Inverts the image. Faint shadows on white paper often 
        look clearer as light strokes on a dark background.
        """
        return cv2.bitwise_not(img)

    def run_ocr(self, img):
        """
        Attempts to read the recovered text using Tesseract.
        """
        # Convert to PIL Image for Tesseract
        pil_img = Image.fromarray(img)
        return pytesseract.image_to_string(pil_img)

    def process_pipeline(self, img, clahe_clip=2.0, light_angle=45, denoise_strength=10, invert=False):
        """
        Runs the full forensic enhancement pipeline.
        """
        # 1. Normalize lighting/contrast
        res = self.apply_clahe(img, clip_limit=clahe_clip)
        
        # 2. Simulate directional light to cast shadows in indentations
        res = self.virtual_lighting(res, angle_deg=light_angle)
        
        # 3. Clean up paper grain noise
        if denoise_strength > 0:
            res = self.denoise(res, strength=denoise_strength)
            
        # 4. Final inversion if requested
        if invert:
            res = self.apply_negative(res)
            
        return res
