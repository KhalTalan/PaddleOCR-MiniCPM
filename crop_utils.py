import os
import cv2
import numpy as np
from paddleocr import PPStructure
from pathlib import Path

def extract_section_12_crop(image_path, output_path=None):
    """
    Extracts Section 12 (Circonstances) from a Constat Amiable image.
    Uses PaddleOCR to find the layout blocks.
    """
    image_path = Path(image_path)
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_crop_section12{image_path.suffix}"
    
    print(f"✂️  Cropping Section 12 for: {image_path.name}")
    
    # 1. Load Image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Error: Could not load image {image_path}")
        return None
        
    h, w = img.shape[:2]
    
    # 2. Run Layout Analysis
    # image_orientation=False usually faster/safer if images are upright
    engine = PPStructure(show_log=False, image_orientation=False)
    result = engine(img)
    
    # 3. Find Anchor Blocks
    # We look for "12. circonstances" or "circonstances"
    header_bbox = None
    footer_bbox = None
    
    content_bboxes = []
    
    for region in result:
        # PPStructure result region keys: 'type', 'bbox', 'res'
        # 'res' contains text recognition results
        bbox = region.get('bbox') # [x1, y1, x2, y2]
        res = region.get('res', [])
        
        # Check text content
        all_text = ""
        if isinstance(res, list):
            # res is list of dicts
            for line in res:
                if isinstance(line, dict):
                    all_text += " " + line.get('text', '').lower()
                elif isinstance(line, str):
                    all_text += " " + line.lower()
        
        # Header detection
        if "circonstances" in all_text and "12" in all_text:
            header_bbox = bbox
            print(f"   📍 Found Header: {bbox}")
        elif "circonstances" in all_text and header_bbox is None:
            # Fallback
            header_bbox = bbox
            print(f"   📍 Found Header (fuzzy): {bbox}")
            
        # Footer detection (end of list)
        if "nombre de cases" in all_text or "croquis" in all_text or "signature" in all_text or "mes observations" in all_text:
            # We want the highest footer that is below the header
            if header_bbox and bbox[1] > header_bbox[1]:
                if footer_bbox is None or bbox[1] < footer_bbox[1]:
                    footer_bbox = bbox
                    print(f"   📍 Found Footer: {bbox} ({all_text[:20]}...)")

        # Collect content blocks that might be the list items
        if header_bbox and bbox[1] > header_bbox[3]:
             content_bboxes.append(bbox)

    # 4. Calculate Crop Coordinates
    if header_bbox is None:
        print("⚠️  Warning: Header '12. Circonstances' not found. Using Center Crop heuristic.")
        # Fallback: Middle 50% width, vertical middle-ish
        crop_x1 = int(w * 0.1)
        crop_x2 = int(w * 0.9)
        crop_y1 = int(h * 0.2)
        crop_y2 = int(h * 0.8)
    else:
        # Y-Coordinates
        crop_y1 = max(0, header_bbox[1] - 20) # Include header
        
        if footer_bbox:
            crop_y2 = min(h, footer_bbox[3] + 20)
        else:
            # Heuristic: Header + 600px or until end
            crop_y2 = min(h, header_bbox[3] + 700)
            
        # X-Coordinates
        # The header bbox usually covers the text column.
        # Checkboxes are to left and right. 
        # Header "12. Circonstances" is usually in the middle column.
        header_cx = (header_bbox[0] + header_bbox[2]) / 2
        header_width = header_bbox[2] - header_bbox[0]
        
        # We assume the section is centered on the header/list
        # The full section width (including checkboxes) is wide relative to the label text.
        # Typically Checkbox A (Left) + Text (Center) + Checkbox B (Right)
        # Let's verify the content blocks width
        
        current_y_limit = crop_y2
        
        # Find min/max X of content blocks within the Y range
        valid_x = []
        for b in content_bboxes:
            if b[1] < current_y_limit:
                 valid_x.append(b[0])
                 valid_x.append(b[2])
        
        if valid_x:
            min_content_x = min(valid_x)
            max_content_x = max(valid_x)
            print(f"   📏 Content Width detected: {min_content_x} to {max_content_x}")
            
            # Add padding for checkboxes (approx 60-80px each side of the text block)
            # If the content detected was just the text labels, we need padding.
            # If extracting just text, result might be narrow.
            crop_x1 = max(0, min_content_x - 85)
            crop_x2 = min(w, max_content_x + 85)
        else:
            # Fallback based on header
            crop_x1 = max(0, header_bbox[0] - 100)
            crop_x2 = min(w, header_bbox[2] + 100)

    # Final sanity check on aspect ratio? 
    # Just save it.
    
    crop_img = img[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]
    
    cv2.imwrite(str(output_path), crop_img)
    print(f"✅ Saved Crop: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()
    
    extract_section_12_crop(args.image_path)
