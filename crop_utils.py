import os
import cv2
import json
import numpy as np
import tempfile
from pathlib import Path
from paddleocr import PaddleOCRVL

# Suppress warnings
os.environ["PPOCR_KEY_WARNING"] = "False"

def extract_section_12_crop(image_path, output_path=None):
    """
    Extracts Section 12 (Circonstances) from a Constat Amiable image.
    Uses PaddleOCRVL layout analysis.
    """
    image_path = Path(image_path)
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_crop_section12{image_path.suffix}"
    
    print(f"✂️  Cropping Section 12 for: {image_path.name}")
    
    # 1. Initialize PaddleOCR-VL (known to work)
    try:
        pipeline = PaddleOCRVL(
            use_layout_detection=True,
            layout_unclip_ratio=1.05,
            layout_nms=False
        )
    except Exception as e:
        print(f"❌ Error initializing PaddleOCRVL: {e}")
        return None

    # 2. Run Layout Analysis
    try:
        output = pipeline.predict(str(image_path))
    except Exception as e:
         print(f"❌ Error running pipeline: {e}")
         return None
         
    if not output:
        print("❌ No output from OCR")
        return None
        
    # Process first page
    res = output[0]
    
    # 3. Extract Layout Blocks (via temporary JSON export for reliability)
    fd, temp_json = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    
    try:
        res.save_to_json(temp_json)
        with open(temp_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    finally:
        if os.path.exists(temp_json):
            os.remove(temp_json)
            
    blocks = data.get('parsing_res_list', [])
    
    # 4. Find Anchor Blocks
    header_bbox = None
    footer_bbox = None
    content_bboxes = []
    
    # Load image for cropping
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ CV2 could not load image: {image_path}")
        return None
    h, w = img.shape[:2]
    
    print(f"   🔍 Analyzing {len(blocks)} layout blocks...")
    
    for block in blocks:
        bbox = block.get('block_bbox') # [x1, y1, x2, y2]
        text = block.get('block_content', '').lower()
        print(f"      - Block: {text[:50]}...") # Debug print
        
        # Header detection
        if "circonstances" in text and "12" in text:
            header_bbox = bbox
            print(f"   📍 Found Header: {bbox}")
        elif "circonstances" in text and header_bbox is None:
            header_bbox = bbox
            # print(f"   📍 Found Header (fuzzy): {bbox}")
            
        # Footer detection
        # Look for end of section signals
        if "nombre de cases" in text or "croquis" in text or "signature" in text or "mes observations" in text:
            if header_bbox and bbox[1] > header_bbox[1]:
                if footer_bbox is None or bbox[1] < footer_bbox[1]:
                    footer_bbox = bbox
                    print(f"   📍 Found Footer: {bbox}")
                    
        # Collect likely list items if header found
        if header_bbox and bbox[1] > header_bbox[3]:
             if footer_bbox is None or bbox[1] < footer_bbox[1]:
                 content_bboxes.append(bbox)

    # 5. Calculate Crop Coordinates
    # Heuristics based on standard Constat form layout
    
    if header_bbox:
         # Vertical Extent
         crop_y1 = max(0, header_bbox[1] - 10)
         
         if footer_bbox:
             crop_y2 = min(h, footer_bbox[3] + 10)
         else:
             crop_y2 = min(h, crop_y1 + 600) # Heuristic height
             
         # Horizontal Extent - Tricky part
         # The "12. Circonstances" title is usually centered above the list.
         # The list items (labels) are centered.
         # Checkboxes are Left and Right of labels.
         
         # Detect width of the text content column
         valid_x = [b[0] for b in content_bboxes] + [b[2] for b in content_bboxes]
         
         if valid_x:
             min_content_x = min(valid_x)
             max_content_x = max(valid_x)
             # Expand generously to capture checkboxes
             # Checkboxes ~40-50px wide, plus margin
             expand_x = 100 
             
             crop_x1 = max(0, min_content_x - expand_x)
             crop_x2 = min(w, max_content_x + expand_x)
             
             print(f"   📏 Content Width: {min_content_x}-{max_content_x} -> Crop: {crop_x1}-{crop_x2}")
         else:
             # Fallback based on Header width
             # Assume header ~30% of form width or ~150px
             # Expand 150px on each side
             crop_x1 = max(0, header_bbox[0] - 150)
             crop_x2 = min(w, header_bbox[2] + 150)
    else:
         print("⚠️ Header not found, using percentiles")
         # Center crop fallback
         crop_x1 = int(w * 0.1)
         crop_x2 = int(w * 0.9)
         crop_y1 = int(h * 0.25)
         crop_y2 = int(h * 0.75)

    # Execute Crop
    crop_img = img[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]
    
    # Save
    cv2.imwrite(str(output_path), crop_img)
    print(f"✅ Saved Crop: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()
    try:
        extract_section_12_crop(args.image_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
