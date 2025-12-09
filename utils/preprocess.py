"""
Image preprocessing utilities for enhanced checkbox and text detection
"""

from PIL import Image, ImageEnhance
import numpy as np
from pathlib import Path


def preprocess_image(image_path, output_dir=None):
    """
    Enhance image for better checkbox and text detection.
    Applies grayscale, contrast boost, and sharpening.
    
    Args:
        image_path: Path to original image
        output_dir: Directory to save enhanced image (default: same as input)
    
    Returns:
        Path to enhanced image
    """
    image_path = Path(image_path)
    
    print(f"🎨 Preprocessing: {image_path.name}")
    
    # Load original image
    original_img = Image.open(image_path).convert('RGB')
    
    # Determine output directory
    if output_dir is None:
        output_dir = image_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Convert to grayscale
    img_gray = original_img.convert('L')
    
    # Step 2: Enhance contrast (50% boost)
    img_contrast = ImageEnhance.Contrast(Image.fromarray(np.array(img_gray))).enhance(1.5)
    
    # Step 3: Enhance sharpness (double sharpness)
    img_sharp = ImageEnhance.Sharpness(img_contrast).enhance(2.0)
    
    # Convert back to RGB
    img_enhanced = img_sharp.convert('RGB')
    
    # Save final enhanced version only
    enhanced_path = output_dir / f"{image_path.stem}_enhanced.jpg"
    img_enhanced.save(enhanced_path, quality=95)
    
    print(f"   ✅ Enhanced: {enhanced_path.name}")
    return str(enhanced_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()
    preprocess_image(args.image_path)
