import cv2
import numpy as np
from pathlib import Path

def extract_section_12_crop(image_path, output_path=None, output_dir=None, debug=False):
    """
    Extracts Section 12 using MANUAL COORDINATES.
    Since Constat forms are standardized, we use percentage-based cropping.
    
    Args:
        image_path: Path to input image
        output_path: Full path for output file (overrides output_dir)
        output_dir: Directory to save crop (uses default filename)
        debug: Save debug image showing crop region
    """
    image_path = Path(image_path)
    
    if output_path is None:
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{image_path.stem}_crop_section12{image_path.suffix}"
        else:
            output_path = image_path.parent / f"{image_path.stem}_crop_section12{image_path.suffix}"
    
    print(f"✂️  Cropping Section 12 (Manual Method): {image_path.name}")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Could not load image")
        return None
    
    h, w = img.shape[:2]
    print(f"   📐 Image size: {w}x{h}")
    
    # STANDARDIZED CONSTAT FORM LAYOUT (percentages):
    # - Top 15%: Header (blue bar "CONSTAT AMIABLE...")
    # - Next 5%: Date/Location fields  
    # - Next 45%: Vehicle details + Section 12 checkboxes
    # - Bottom 35%: Signatures, observations
    
    # Section 12 is typically:
    # - Vertical: 20-65% of height (below header, above signatures)
    # - Horizontal: CENTER 40% of width (between yellow columns)
    
    # Vertical crop (Section 12 is in middle portion)
    crop_y1 = int(h * 0.20)  # Start after header/date (20%)
    crop_y2 = int(h * 0.70)  # End before signatures (70%)
    
    # Horizontal crop (Section 12 is center column)
    crop_x1 = int(w * 0.35)  # Left edge of center column
    crop_x2 = int(w * 0.65)  # Right edge of center column
    
    print(f"   ✂️  Crop region: x=[{crop_x1}, {crop_x2}] ({crop_x2-crop_x1}px), y=[{crop_y1}, {crop_y2}] ({crop_y2-crop_y1}px)")
    
    # Execute crop
    crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Save
    cv2.imwrite(str(output_path), crop_img)
    print(f"✅ Saved: {output_path}")
    
    if debug:
        # Save annotated version showing crop region
        debug_img = img.copy()
        cv2.rectangle(debug_img, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 3)
        debug_path = image_path.parent / f"{image_path.stem}_crop_debug.jpg"
        cv2.imwrite(str(debug_path), debug_img)
        print(f"   🔍 Debug image: {debug_path}")
    
    return str(output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--debug", action="store_true", help="Save debug image showing crop region")
    args = parser.parse_args()
    extract_section_12_crop(args.image_path, debug=args.debug)
