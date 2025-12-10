"""
Utility to crop Section 12 (Circonstances) from Constat Amiable forms.
Uses Qwen VLM to detect the exact bounding box coordinates.
"""

from PIL import Image
from pathlib import Path
import re


def ask_qwen_for_bbox(model, processor, image_path):
    """
    Ask Qwen to identify Section 12 bounding box coordinates.
    Returns (left, top, right, bottom) or None if failed.
    """
    prompt = """Look at this French Constat Amiable accident report form.

Find Section 12 "CIRCONSTANCES" which contains the checkbox grid (17 rows) with:
- Blue column on the left (Vehicle A)
- Yellow column on the right (Vehicle B)

Return ONLY the bounding box coordinates in this exact format:
LEFT,TOP,RIGHT,BOTTOM

Where:
- LEFT = x-coordinate of the left edge of Section 12
- TOP = y-coordinate of the top edge (where "12. CIRCONSTANCES" header starts)
- RIGHT = x-coordinate of the right edge
- BOTTOM = y-coordinate of the bottom edge (after the last checkbox row and manual count numbers)

Example response: 100,250,700,900

Now provide the coordinates for this image:"""
    
    # Build message
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt}
        ]
    }]
    
    try:
        # Apply chat template
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        # Generate
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,  # Low temp for precise coordinates
            top_p=0.9
        )
        
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"   Qwen response: {output_text}")
        
        # Parse coordinates from response
        # Look for pattern: number,number,number,number
        match = re.search(r'(\d+),\s*(\d+),\s*(\d+),\s*(\d+)', output_text)
        if match:
            left, top, right, bottom = map(int, match.groups())
            return left, top, right, bottom
        
        return None
        
    except Exception as e:
        print(f"   ⚠️ Qwen detection failed: {e}")
        return None


def extract_section_12_crop(image_path, output_dir=None, model=None, processor=None):
    """
    Extract Section 12 using VLM detection.
    
    Args:
        image_path: Path to the full Constat image
        output_dir: Directory to save crop (default: same as input)
        model: Qwen model (if None, falls back to percentage-based)
        processor: Qwen processor (if None, falls back to percentage-based)
    
    Returns:
        Path to cropped image, or None if failed
    """
    image_path = Path(image_path)
    
    try:
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        
        # Try VLM detection if model provided
        if model is not None and processor is not None:
            print(f"   Using Qwen VLM for bbox detection...")
            bbox = ask_qwen_for_bbox(model, processor, image_path)
            
            if bbox:
                left, top, right, bottom = bbox
                print(f"   VLM detected: L={left}, T={top}, R={right}, B={bottom}")
            else:
                print("   ⚠️ VLM detection failed, using fallback")
                left = int(width * 0.01)
                right = int(width * 0.99)
                top = int(height * 0.40)
                bottom = int(height * 0.70)
        else:
            # Fallback to percentage-based
            print("   No VLM provided, using fallback percentages")
            left = int(width * 0.01)
            right = int(width * 0.99)
            top = int(height * 0.40)
            bottom = int(height * 0.70)
        
        print(f"   Final crop: L={left}, T={top}, R={right}, B={bottom}")
        
        # Crop
        cropped = img.crop((left, top, right, bottom))
        
        # Determine output path
        if output_dir is None:
            output_dir = image_path.parent
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


if __name__ == "__main__":
    import sys
    print("This module requires Qwen model to be loaded.")
    print("Use from test_qwen_twostep.py instead.")
