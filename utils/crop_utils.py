"""
Utility to crop Section 12 (Circonstances) from Constat Amiable forms.
Uses color detection for all bounds:
- Horizontal: scan from center to find blue (left) and yellow (right)
- Vertical top: find first row with both colors
- Vertical bottom: detect when blue/yellow columns become narrower
"""

from PIL import Image
import numpy as np
from pathlib import Path


def is_blue_pixel(r, g, b):
    """Check if pixel is blue (Vehicle A color)"""
    return b > 120 and b > r and b > g and r < 150

def is_yellow_pixel(r, g, b):
    """Check if pixel is yellow (Vehicle B color)"""
    return r > 180 and g > 150 and b < 120


def find_horizontal_bounds(img_array, sample_rows):
    """Find left (blue) and right (yellow) by scanning from center."""
    height, width = img_array.shape[:2]
    center_x = width // 2
    
    left_bounds = []
    right_bounds = []
    
    for row in sample_rows:
        if row >= height:
            continue
            
        # Scan LEFT from center to find blue
        for x in range(center_x, 0, -1):
            r, g, b = img_array[row, x]
            if is_blue_pixel(r, g, b):
                left_bounds.append(x)
                break
        
        # Scan RIGHT from center to find yellow
        for x in range(center_x, width):
            r, g, b = img_array[row, x]
            if is_yellow_pixel(r, g, b):
                right_bounds.append(x)
                break
    
    if left_bounds and right_bounds:
        return min(left_bounds), max(right_bounds)
    return None, None


def measure_color_width(img_array, row, left, right):
    """Measure the width of blue and yellow regions in a row."""
    width = img_array.shape[1]
    
    blue_pixels = 0
    yellow_pixels = 0
    
    # Count blue pixels near left boundary
    for x in range(max(0, left - 30), min(width, left + 60)):
        r, g, b = img_array[row, x]
        if is_blue_pixel(r, g, b):
            blue_pixels += 1
    
    # Count yellow pixels near right boundary
    for x in range(max(0, right - 60), min(width, right + 30)):
        r, g, b = img_array[row, x]
        if is_yellow_pixel(r, g, b):
            yellow_pixels += 1
    
    return blue_pixels, yellow_pixels


def find_vertical_bounds(img_array, left, right):
    """
    Find top and bottom:
    - Top: first row with both blue and yellow
    - Bottom: detect when color columns become narrower (Section 12 ends)
    """
    height, width = img_array.shape[:2]
    
    # Find top: first row with both colors
    top = None
    for row in range(height):
        has_blue = False
        has_yellow = False
        
        for x in range(max(0, left - 20), min(width, left + 50)):
            r, g, b = img_array[row, x]
            if is_blue_pixel(r, g, b):
                has_blue = True
                break
        
        for x in range(max(0, right - 50), min(width, right + 20)):
            r, g, b = img_array[row, x]
            if is_yellow_pixel(r, g, b):
                has_yellow = True
                break
        
        if has_blue and has_yellow:
            top = row
            break
    
    if top is None:
        return None, None
    
    # Measure typical color width in Section 12 (sample from top area)
    sample_widths = []
    for row in range(top + 10, min(top + 100, height)):
        blue_w, yellow_w = measure_color_width(img_array, row, left, right)
        if blue_w > 5 and yellow_w > 5:
            sample_widths.append((blue_w, yellow_w))
    
    if not sample_widths:
        return top, None
    
    # Calculate typical width
    avg_blue_width = sum(w[0] for w in sample_widths) / len(sample_widths)
    avg_yellow_width = sum(w[1] for w in sample_widths) / len(sample_widths)
    
    print(f"   Typical widths: blue={avg_blue_width:.0f}, yellow={avg_yellow_width:.0f}")
    
    # Find bottom: where width drops significantly (< 10% of typical)
    # Need 15 consecutive narrow rows to confirm end of section
    bottom = None
    consecutive_narrow = 0
    
    for row in range(top + 50, height):
        blue_w, yellow_w = measure_color_width(img_array, row, left, right)
        
        # Check if width dropped significantly (must be BOTH narrow)
        blue_narrow = blue_w < avg_blue_width * 0.1
        yellow_narrow = yellow_w < avg_yellow_width * 0.1
        
        if blue_narrow and yellow_narrow:
            consecutive_narrow += 1
            if consecutive_narrow >= 15:
                bottom = row - 15  # Go back to where it started narrowing
                break
        else:
            consecutive_narrow = 0
    
    if bottom is None:
        # Fallback: use last row with good width
        for row in range(height - 1, top, -1):
            blue_w, yellow_w = measure_color_width(img_array, row, left, right)
            if blue_w > avg_blue_width * 0.5 and yellow_w > avg_yellow_width * 0.5:
                bottom = row
                break
    
    return top, bottom


def extract_section_12_crop(image_path, output_dir=None):
    """
    Extract Section 12 (Circonstances) from a Constat Amiable form.
    
    Uses color detection for all bounds:
    - Left: scan from center to find blue
    - Right: scan from center to find yellow  
    - Top: first row with both colors
    - Bottom: detect when color columns become narrower
    
    Args:
        image_path: Path to the full Constat image
        output_dir: Directory to save crop (default: same as input)
    
    Returns:
        Path to cropped image, or None if failed
    """
    image_path = Path(image_path)
    
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        width, height = img.size
        
        # First pass: find rough horizontal bounds using middle rows
        sample_rows = [int(height * p) for p in [0.45, 0.50, 0.55, 0.60]]
        left, right = find_horizontal_bounds(img_array, sample_rows)
        
        if left is None or right is None:
            print("   ⚠️ Horizontal detection failed, using fallback")
            left = int(width * 0.01)
            right = int(width * 0.99)
        else:
            print(f"   Horizontal: L={left}, R={right}")
        
        # Second pass: find vertical bounds
        top, bottom = find_vertical_bounds(img_array, left, right)
        
        if top is None:
            print("   ⚠️ Top detection failed, using fallback")
            top = int(height * 0.40)
        else:
            print(f"   Top: {top}")
            
        if bottom is None:
            print("   ⚠️ Bottom detection failed, using fallback")
            bottom = int(height * 0.70)
        else:
            print(f"   Bottom: {bottom}")
        
        # Small margins
        left = max(0, left - 2)
        right = min(width, right + 2)
        top = max(0, top - 2)
        bottom = min(height, bottom + 2)
        
        print(f"   Final crop: L={left}, T={top}, R={right}, B={bottom}")
        
        # Crop
        cropped = img.crop((left, top, right, bottom))
        
        # Determine output path - use dedicated output directory
        if output_dir is None:
            # Default: save to output_crops/ at project root
            output_dir = Path(__file__).parent.parent / "output/crops"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        crop_path = output_dir / f"{image_path.stem}_crop_section12.jpg"
        cropped.save(crop_path, quality=95)
        
        print(f"✅ Saved: {crop_path}")
        return str(crop_path)
        
    except Exception as e:
        print(f"❌ Cropping failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_and_enhance(image_path, output_dir=None):
    """
    Crop Section 12 AND enhance the cropped image.
    Returns path to the enhanced crop.
    """
    from preprocess import preprocess_image
    
    crop_path = extract_section_12_crop(image_path, output_dir)
    
    if crop_path:
        # Enhance the cropped image
        enhanced_path = preprocess_image(crop_path, output_dir)
        return enhanced_path
    
    return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        enhance = "--enhance" in sys.argv or "-e" in sys.argv
        image_path = [arg for arg in sys.argv[1:] if not arg.startswith("-")][0]
        
        if enhance:
            extract_and_enhance(image_path)
        else:
            extract_section_12_crop(image_path)
    else:
        print("Usage: python crop_utils.py <image_path> [--enhance]")
        print("  --enhance, -e: Also enhance the cropped image")
