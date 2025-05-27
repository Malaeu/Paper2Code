#!/usr/bin/env python3
"""
Vision annotator for PDF images: detects tables/figures, extracts chart types,
font families, and color palettes from images.
"""
import cv2
import json
import sys
import pathlib
import logging
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vision_annotator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("vision_annotator")

# Dictionary to store detected features
results = []

def detect_image_type(image_path):
    """
    Detects if an image contains a table, figure, or neither.
    In a production environment, this would use a trained model like YOLO.
    For this example implementation, we'll use a simple heuristic approach.
    """
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return "none"
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple heuristic: detect horizontal and vertical lines for tables
        # This is a very basic approach and would be replaced with a proper model
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None and len(lines) > 20:
            # Count horizontal vs vertical lines
            h_lines = 0
            v_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) > abs(y2 - y1):  # Horizontal line
                    h_lines += 1
                else:  # Vertical line
                    v_lines += 1
            
            # If we have a good mix of horizontal and vertical lines, it's likely a table
            if h_lines > 5 and v_lines > 5 and h_lines/len(lines) > 0.2 and v_lines/len(lines) > 0.2:
                return "table"
            else:
                return "figure"
        
        # Check for image complexity (figures typically have more varied content)
        std_dev = np.std(gray)
        if std_dev > 50:  # Higher standard deviation suggests a figure with varied content
            return "figure"
        
        return "none"  # Default case
    
    except Exception as e:
        logger.error(f"Error detecting image type for {image_path}: {e}")
        return "none"

def detect_chart_type(image_path, image_type):
    """
    Detects the type of chart in a figure.
    In production, this would use a trained classifier.
    """
    if image_type != "figure":
        return "N/A"
    
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return "unknown"
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple heuristic: look for circles (pie charts, scatter plots)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=100)
        
        # Detect edges for contour analysis
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count what appear to be bars
        potential_bars = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h != 0 else 0
            
            # Typical bar characteristics
            if (0.1 < aspect_ratio < 10) and (w > 10 or h > 10):
                potential_bars += 1
        
        # Simple classification logic
        if circles is not None:
            if potential_bars > 10:
                return "scatter_plot"
            else:
                return "pie_chart"
        elif potential_bars > 5:
            return "bar_chart"
        else:
            return "line_chart"  # Default guess for figures
            
    except Exception as e:
        logger.error(f"Error detecting chart type for {image_path}: {e}")
        return "unknown"

def extract_color_palette(image_path):
    """
    Extracts dominant colors from the image to determine the color palette.
    """
    try:
        # Load image with PIL
        img = Image.open(image_path)
        img = img.convert("RGB")
        img = img.resize((100, 100))  # Resize for faster processing
        
        # Convert to numpy array
        img_array = np.array(img)
        pixels = img_array.reshape(-1, 3)
        
        # Simple color clustering (in production would use k-means or similar)
        # For this example, we'll just pick some distinct colors
        unique_colors = set()
        for pixel in pixels[::100]:  # Sample every 100th pixel
            r, g, b = pixel
            # Round to nearest 50 to reduce color space
            r = round(r / 50) * 50
            g = round(g / 50) * 50
            b = round(b / 50) * 50
            
            if r + g + b > 30:  # Skip near-black colors
                unique_colors.add((r, g, b))
        
        # Convert to hex format
        palette = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in unique_colors]
        
        # Limit to top 5 colors
        return palette[:5]
    
    except Exception as e:
        logger.error(f"Error extracting color palette from {image_path}: {e}")
        return ["#000000"]  # Default black

def detect_font_family(image_path):
    """
    Detects the primary font family used in the image.
    In production, this would use OCR and font recognition models.
    For this example, we'll return a reasonable guess for academic papers.
    """
    # Academic papers typically use one of these fonts
    common_fonts = ["Times New Roman", "Arial", "Computer Modern", "Helvetica"]
    
    # In a real implementation, we would use OCR and font detection
    # For now, just return a reasonable default
    return "Computer Modern"  # Most common in academic papers

def process_image(image_path):
    """Process a single image and extract its features."""
    try:
        path = pathlib.Path(image_path)
        
        # Check if this is a blank page
        # (A proper blank page detector would analyze the pixel distribution)
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        # Simple blank page detection - check if it's mostly white
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) > 240 and np.std(gray) < 10:
            logger.info(f"Detected blank page: {image_path}")
            return {
                "page": path.stem,
                "object": "none",
                "blank_page": True,
                "chart_type": "N/A",
                "font_family": "N/A",
                "palette": []
            }
        
        # Detect image type (table, figure, or none)
        image_type = detect_image_type(image_path)
        
        # Detect chart type if it's a figure
        chart_type = detect_chart_type(image_path, image_type)
        
        # Extract color palette
        palette = extract_color_palette(image_path)
        
        # Detect font family
        font_family = detect_font_family(image_path)
        
        # Compile results
        result = {
            "page": path.stem,
            "object": image_type,
            "chart_type": chart_type if image_type == "figure" else "N/A",
            "font_family": font_family,
            "palette": palette
        }
        
        logger.info(f"Processed {image_path}: {image_type}, {chart_type}")
        return result
    
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None

def main():
    # Check if any image paths were provided
    if len(sys.argv) < 2:
        logger.error("No image paths provided. Usage: python vision_annotator.py image1.jpg image2.jpg ...")
        sys.exit(1)
    
    # Process each image
    for img_path in sys.argv[1:]:
        result = process_image(img_path)
        if result:
            results.append(result)
    
    # Output JSON to stdout
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()