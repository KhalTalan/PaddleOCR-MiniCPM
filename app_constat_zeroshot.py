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

# ================== LOAD MODELS ==================

def load_paddleocr_vl():
    """Load PaddleOCR-VL using official API"""
    print("📦 Loading PaddleOCR-VL...")
    from paddleocr import PaddleOCRVL
    pipeline = PaddleOCRVL()
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
    Comprehensive zero-shot prompt with checklist-based checkbox verification.
    Uses OCR to understand box meanings, requires visual verification.
    """
    ocr_content = "\n".join(ocr_texts)
    
    prompt = f"""You are analyzing a French Constat Amiable d'Accident Automobile (official accident report form).

OCR EXTRACTED TEXT:
{ocr_content}

---

⚠️⚠️⚠️ CRITICAL INSTRUCTIONS FOR SECTION 12 (CIRCUMSTANCES) ⚠️⚠️⚠️

Section 12 lists 17 possible circumstances boxes for each vehicle.
The OCR above shows the TEXT LABELS for these boxes (e.g., "8 heurtait l'arrière", "12 virait à droite").

IMPORTANT: Just because the OCR extracted the text label DOES NOT mean the box is checked!

YOUR TASK FOR SECTION 12:
For EACH of the 17 boxes, visually inspect the IMAGE and determine if it's checked or not.

USE THIS FORMAT for Section 4 (CIRCUMSTANCES):

**Vehicle A (Left Column):**
Go through boxes 1-17. For each box, look at the checkbox square:
- If you see a WHITE/EMPTY box → ☐ NOT CHECKED
- If you see a MARK inside (✓, X, ☑, or any dark fill) → ☑ CHECKED

Only list the boxes that are CHECKED. Example:
"Vehicle A: Box 8 ☑, Box 12 ☑" 
OR if none: "Vehicle A: No boxes checked"

**Vehicle B (Right Column):**
Do the same for the right side boxes 1-17.

DO NOT simply list boxes whose text appeared in the OCR!
You MUST visually verify each checkbox is marked!

---

YOUR ANALYSIS FORMAT:

---
CONSTAT AMIABLE ANALYSIS

1. ACCIDENT DETAILS
Date: [from Section 1]
Time: [from Section 1]
Location: [from Section 2]
Injuries: [Yes/No from Section 3]
Other damage: [Yes/No from Section 4]
Witnesses: [from Section 5]

2. VEHICLE A (Left side)
Driver: [Name, DOB from Section 9]
Address: [Full address]
Phone: [Phone number]
Vehicle: [Make/Model from Section 7]
Registration: [Number]
Insurance: [Company from Section 8, Contract #]
License: [Category, Number, Valid until]
Damage: [from Section 11]
Observation: "[EXACT quote from Section 14]" - [Does this BLAME other driver? Yes/No + explanation]

3. VEHICLE B (Right side)
[Same structure as Vehicle A]

4. CIRCUMSTANCES (Section 12) **VISUAL VERIFICATION REQUIRED**

Vehicle A: [ONLY list checked boxes - e.g., "Box 8 ☑, Box 12 ☑" OR "No boxes checked"]
Vehicle B: [ONLY list checked boxes - e.g., "Box 2 ☑" OR "No boxes checked"]

Note: There are 17 total boxes. The OCR shows all labels, but you must verify visually which are marked.

5. RECONSTRUCTION
[Based ONLY on the boxes you verified as checked above, reconstruct what happened]

6. FAULT ANALYSIS
[Apply French liability rules based on the verified checked boxes]

7. SUMMARY
[Brief conclusion: date, location, what happened, who's at fault]
---

GENERAL RULES:
- Be factual and concise
- Write "Not legible" if you can't read something
- DO NOT invent information
- Quote observations exactly
- Base fault analysis ONLY on visually verified checked boxes

Now analyze the provided Constat image."""
    
    return prompt


def analyze_constat_zeroshot(test_image_path, pipeline=None, model=None, tokenizer=None):
    """
    Analyze a Constat image using zero-shot learning with detailed prompt.
    No example image needed. Uses PaddleOCR-VL for better OCR.
    """
    # ================== STEP 1: OCR PROCESSING (PaddleOCR-VL) ==================
    
    if pipeline is None:
        pipeline = load_paddleocr_vl()
    
    print("\n🎯 Processing image OCR...")
    test_image = Image.open(test_image_path).convert('RGB')
    test_ocr_texts = extract_ocr_text_vl(pipeline, test_image_path)
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
