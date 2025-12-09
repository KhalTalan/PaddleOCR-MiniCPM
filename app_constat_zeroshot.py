#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-Shot Learning for French Constat Analysis with PaddleOCR + MiniCPM-V-2_6
Uses detailed prompt without example image to analyze French accident reports.
"""

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import os
import sys
import gc
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ================== CONFIG ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path.home() / ".cache" / "dual_vlm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(CACHE_DIR / "hf")
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR / "transformers")

# Get Hugging Face token from environment
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    print("⚠️  Warning: HF_TOKEN not found in .env file")
    print("   The MiniCPM-V-2_6 model is gated and requires authentication.")
    print("   Please create a .env file with: HF_TOKEN=your_token_here")

print(f"🚀 Device: {DEVICE}")
if HF_TOKEN:
    print(f"🔑 HF Token loaded: {HF_TOKEN[:10]}...")


# ================== IMAGE PREPROCESSING ==================

def preprocess_image_for_detection(image_path, save_debug=True):
    """
    Enhance image for better checkbox and text detection.
    Applies multiple enhancement techniques and saves debug versions.
    
    Args:
        image_path: Path to original image
        save_debug: Whether to save debug images showing enhancements
    
    Returns:
        Path to enhanced image
    """
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    from pathlib import Path
    
    print(f"🎨 Preprocessing image: {Path(image_path).name}")
    
    # Load original image
    original_img = Image.open(image_path).convert('RGB')
    
    # Create output directory
    output_dir = Path(__file__).parent / "output" / Path(image_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create debug subdirectory for preprocessing images
    debug_dir = output_dir / "preprocessing_debug"
    if save_debug:
        debug_dir.mkdir(exist_ok=True)
        # Save original for comparison
        original_img.save(debug_dir / "00_original.jpg", quality=95)
    
    # Step 1: Convert to grayscale for better contrast
    img_gray = original_img.convert('L')
    if save_debug:
        img_gray.save(debug_dir / "01_grayscale.jpg", quality=95)
    
    # Step 2: Enhance contrast
    enhancer = ImageEnhance.Contrast(Image.fromarray(np.array(img_gray)))
    img_contrast = enhancer.enhance(1.5)  # Increase contrast by 50%
    if save_debug:
        img_contrast.save(debug_dir / "02_contrast_enhanced.jpg", quality=95)
    
    # Step 3: Enhance sharpness to make checkmarks more visible
    enhancer_sharp = ImageEnhance.Sharpness(img_contrast)
    img_sharp = enhancer_sharp.enhance(2.0)  # Double sharpness
    if save_debug:
        img_sharp.save(debug_dir / "03_sharpened.jpg", quality=95)
    
    # Step 4: Apply adaptive thresholding for checkbox detection
    # Convert to numpy for advanced processing
    img_array = np.array(img_sharp)
    
    # Simple adaptive threshold (lighter background, darker marks)
    # This makes checkmarks stand out more
    threshold_value = np.mean(img_array) - 10  # Slightly below mean
    img_binary = np.where(img_array < threshold_value, 0, 255).astype(np.uint8)
    img_threshold = Image.fromarray(img_binary)
    
    if save_debug:
        img_threshold.save(debug_dir / "04_threshold.jpg", quality=95)
    
    # Step 5: Final enhanced version (convert back to RGB for compatibility)
    # Use the sharpened + contrasted version, not the binary threshold
    # (threshold is just for debug visualization)
    img_enhanced = img_sharp.convert('RGB')
    
    # Save enhanced version
    enhanced_path = output_dir / "enhanced_image.jpg"
    img_enhanced.save(enhanced_path, quality=95)
    
    print(f"   ✅ Enhanced image saved: output/{Path(image_path).stem}/enhanced_image.jpg")
    if save_debug:
        print(f"   📊 Debug images saved in: output/{Path(image_path).stem}/preprocessing_debug/")
    
    return str(enhanced_path)


# ================== LOAD MODELS ==================

def load_paddleocr_vl():
    """Load PaddleOCR-VL using official API"""
    print("📦 Loading PaddleOCR-VL...")
    from paddleocr import PaddleOCRVL
    pipeline = PaddleOCRVL(
            # Expand boxes slightly to capture checkbox borders
            layout_unclip_ratio=1.05,
            # Don't merge nearby boxes
            layout_nms=False,
            # Keep layout detection for structured parsing
            use_layout_detection=True,
    )
    print("✅ PaddleOCR-VL loaded")
    return pipeline


def extract_ocr_text_vl(pipeline, image_path, save_debug=True):
    """Extract text using PaddleOCR-VL and return markdown formatted text"""
    print(f"🔍 OCR-VL: {Path(image_path).name}")
    
    # Use PaddleOCR-VL pipeline
    output = pipeline.predict(str(image_path))
    markdown_content = ""
    
    # Save debug outputs (JSON and Markdown)
    if save_debug:
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        for res in output:
            # Create subdirectory for this image
            image_name = Path(image_path).stem
            image_output_dir = output_dir / image_name
            image_output_dir.mkdir(exist_ok=True)
            
            # Save JSON and Markdown for debugging
            res.save_to_json(save_path=str(image_output_dir / "paddleocr_vl.json"))
            res.save_to_markdown(save_path=str(image_output_dir / "paddleocr_vl.md"))
            print(f"   💾 Debug files saved in: output/{image_name}/")
    
    # Read the markdown file that was just saved
    if save_debug:
        markdown_file = image_output_dir / "paddleocr_vl.md"
        if markdown_file.exists():
            markdown_content = markdown_file.read_text(encoding='utf-8')
    
    # Convert markdown to list of text blocks (split by newlines to preserve structure)
    texts = [line.strip() for line in markdown_content.split('\n') if line.strip()]
    
    print(f"   Found {len(texts)} text blocks (from markdown)")
    return texts


def load_minicpm():
    """Load MiniCPM-V-2_6 model"""
    print("📦 Loading MiniCPM-V-2_6...")
    model_name = 'openbmb/MiniCPM-V-2_6'
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=str(CACHE_DIR),
        token=HF_TOKEN
    )
    
    if DEVICE == "cuda":
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16,
            cache_dir=str(CACHE_DIR),
            token=HF_TOKEN
        )
        model = model.eval().cuda()
    else:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.float32,
            cache_dir=str(CACHE_DIR),
            token=HF_TOKEN
        )
        model = model.eval()
    
    print("✅ MiniCPM-V-2_6 loaded")
    return model, tokenizer


def build_zero_shot_prompt(ocr_texts):
    """
    Direct, action-oriented prompt that requests analysis immediately.
    Removes instructional language that confuses the model.
    """
    ocr_content = "\n".join(ocr_texts)
    
    prompt = f"""Analyze this French accident report (Constat Amiable) and provide the structured output below.

OCR TEXT FROM IMAGE:
{ocr_content}

---

OUTPUT THE FOLLOWING 7 SECTIONS:

**1. ACCIDENT DETAILS**
Date: [from image]
Time: [from image]
Location: [from image]
Injuries: Yes/No
Other damage: Yes/No
Witnesses: [if any]

**2. VEHICLE A (Left side)**
Driver: [name, DOB]
Address: [full address]
Phone: [number]
Vehicle: [make/model]
Registration: [number]
Insurance: [company, contract #]
Damage: [description]
Observation: "[exact quote from driver]" - [Does this blame the other driver?]

**3. VEHICLE B (Right side)**
[Same details as Vehicle A]

**4. CIRCUMSTANCES - Section 12 Checkboxes**

CRITICAL: Look at the actual IMAGE in Section 12 (middle of form). 
The OCR above lists ALL 17 box labels, but you must VISUALLY check which boxes have checkmarks.

For each vehicle, inspect boxes 1-17:
- EMPTY box (☐) = white/light = NOT checked
- MARKED box (☑/✓/X) = has mark inside = IS checked

Vehicle A checked boxes: [list only numbers with visible marks, e.g., "8, 14" or "None"]
Vehicle B checked boxes: [list only numbers with visible marks, e.g., "12" or "None"]

**5. RECONSTRUCTION**
[What happened based on the checked boxes above]

**6. FAULT ANALYSIS**
[Who is at fault and why, based on French traffic law and checked boxes]

**7. SUMMARY**
[2-3 sentences: date, location, accident description, fault conclusion]

---

IMPORTANT REMINDERS:
- The OCR lists all box labels (1-17) but this doesn't mean they're checked
- Visually verify each checkbox in the image - most will be EMPTY
- Only list box numbers where you see an actual checkmark/X/fill
- Quote driver observations exactly
- Base all conclusions on visually verified checkboxes

NOW PROVIDE THE ANALYSIS:"""
    
    return prompt


def analyze_constat_zeroshot(test_image_path, pipeline=None, model=None, tokenizer=None):
    """
    Analyze a Constat image using zero-shot learning with detailed prompt.
    No example image needed. Uses PaddleOCR-VL for better OCR.
    Applies image preprocessing for better checkbox and text detection.
    """
    # ================== STEP 0: IMAGE PREPROCESSING ==================
    # Enhance image for better checkbox and text detection
    enhanced_image_path = preprocess_image_for_detection(test_image_path, save_debug=True)
    
    # ================== STEP 1: OCR PROCESSING (PaddleOCR-VL) ==================
    
    if pipeline is None:
        pipeline = load_paddleocr_vl()
    
    print("\n🎯 Processing image OCR...")
    # Use ENHANCED image for OCR
    test_image = Image.open(enhanced_image_path).convert('RGB')
    test_ocr_texts = extract_ocr_text_vl(pipeline, enhanced_image_path)
    test_prompt = build_zero_shot_prompt(test_ocr_texts)
    
    # FREE GPU MEMORY: Delete OCR and clear cache
    print("🧹 Clearing OCR from memory...")
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ================== STEP 2: VLM INFERENCE (MiniCPM-V-2_6) ==================
    
    if model is None or tokenizer is None:
        model, tokenizer = load_minicpm()
    
    print("\n🤖 Running zero-shot inference...")
    
    # ZERO-SHOT: Only the test image, no example
    msgs = [
        {'role': 'user', 'content': [test_image, test_prompt]}
    ]
    
    try:
        answer = model.chat(
            image=None,  # Image is in msgs
            msgs=msgs,
            tokenizer=tokenizer
        )
    except Exception as e:
        import traceback
        answer = f"Error during inference: {e}\n\nTraceback:\n{traceback.format_exc()}"
    
    return {
        "test_image": str(test_image_path),
        "test_ocr_texts": test_ocr_texts,
        "analysis": answer
    }


def main():
    """CLI entry point"""
    if len(sys.argv) < 2:
        print("\n❌ Usage: python app_constat_zeroshot.py <test_image>")
        print("\nThis script uses zero-shot learning to analyze French Constats.")
        print("No example image needed - uses detailed prompt only.")
        print("\nExample:")
        print("   python app_constat_zeroshot.py images/constat.jpg")
        sys.exit(1)
    
    test_image_path = sys.argv[1]
    
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        sys.exit(1)
    
    # Run analysis
    result = analyze_constat_zeroshot(test_image_path)
    
    # Display results
    print("\n" + "=" * 70)
    print("📊 RÉSULTATS DE L'ANALYSE (ZERO-SHOT)")
    print("=" * 70)
    
    print(f"\n📝 Texte extrait ({len(result['test_ocr_texts'])} blocs):")
    for i, text in enumerate(result['test_ocr_texts'][:5], 1):
        preview = text[:80] + "..." if len(text) > 80 else text
        print(f"   {i}. {preview}")
    
    if len(result['test_ocr_texts']) > 5:
        print(f"   ... et {len(result['test_ocr_texts']) - 5} autres")
    
    print("\n🤖 Analyse structurée:")
    print("-" * 70)
    print(result['analysis'])
    print("-" * 70)
    
    
    # Save results to organized output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectory for test image
    test_image_name = Path(test_image_path).stem
    test_output_dir = output_dir / test_image_name
    test_output_dir.mkdir(exist_ok=True)
    
    # Save analysis result
    output_path = test_output_dir / "analysis_zeroshot.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result['analysis'])
    print(f"\n💾 Analyse sauvegardée: output/{test_image_name}/analysis_zeroshot.txt")
    
    # Save OCR output
    ocr_output_path = test_output_dir / "ocr_text.txt"
    with open(ocr_output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(result['test_ocr_texts']))
    print(f"💾 OCR sauvegardé: output/{test_image_name}/ocr_text.txt")
    
    print("✅ Terminé!")


if __name__ == "__main__":
    main()
