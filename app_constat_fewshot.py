#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-Shot Learning for French Constat Analysis with PaddleOCR + MiniCPM-V-2_6
Uses one-shot example to guide the model in interpreting French accident reports.
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

# Paths to few-shot example
EXAMPLE_IMAGE_PATH = Path(__file__).parent / "example_constat.png"
EXPECTED_ANSWER_PATH = Path(__file__).parent / "expected_answer_constat.txt"

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
            attn_implementation='sdpa',  # or 'flash_attention_2' if available
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
    
    # Convert markdown to list of text blocks (split by double newlines to preserve structure)
    texts = [line.strip() for line in markdown_content.split('\n') if line.strip()]
    
    print(f"   Found {len(texts)} text blocks (from markdown)")
    return texts


def build_training_prompt(ocr_texts):
    """Build detailed training prompt for the example with checkbox detection instructions"""
    ocr_content = "\n".join(ocr_texts)
    
    prompt = f"""You are analyzing a French Constat Amiable d'Accident Automobile (official accident report).

OCR DATA:
{ocr_content}

CRITICAL INSTRUCTION FOR SECTION 12 (CIRCUMSTANCES):
You MUST visually inspect the IMAGE to see which boxes are CHECKED (✓, X, ☑).
- The OCR lists all 17 box labels, but most boxes are EMPTY (☐)
- ONLY report box numbers that have a visible checkmark inside
- If no boxes are checked, write "No boxes checked"
- DO NOT assume or guess - use your vision to verify

Analyze this image and extract information in this structured format:

1. ACCIDENT DETAILS: Date, Time, Location, Injuries (yes/no), Other damage (yes/no), Witnesses

2. VEHICLE A (Left): Driver name, Address, Vehicle (make/model), Insurance company and number, License details, Damage description, Driver's observation (quote exactly and note if it blames the other driver)

3. VEHICLE B (Right): Same structure as Vehicle A

4. CIRCUMSTANCES (Section 12): List ONLY the box numbers that are visually CHECKED for each vehicle

5. RECONSTRUCTION: Based on checked boxes and damage, describe what happened

6. FAULT ANALYSIS: Apply French liability rules based on checked circumstances

7. SUMMARY: Brief 1-2 sentence summary

Be concise and evidence-based. Quote driver observations verbatim."""
    
    return prompt


def build_test_prompt(ocr_texts):
    """
    Prompt for the NEW test image with checkbox detection emphasis.
    Includes strong constraints to prevent bleeding from the example.
    """
    ocr_content = "\n".join(ocr_texts)
    
    prompt = f"""Analyze this NEW French Constat Amiable.

⚠️ CRITICAL INSTRUCTIONS:
1. This is a COMPLETELY DIFFERENT accident case from the previous example
2. IGNORE all names, dates, and details from the previous example
3. Use ONLY the information visible in THIS NEW image and OCR text below

SECTION 12 (CIRCUMSTANCES) - VISUAL VERIFICATION REQUIRED:
- Look at THIS new image carefully
- ONLY list box numbers with visible checkmarks (✓, X, ☑)
- Most boxes will be EMPTY (☐) - do not list them
- If unsure, state "No boxes checked" - do NOT guess

NEW OCR TEXT:
{ocr_content}

Provide the analysis in the same 7-section format as the example, but using ONLY the data from this new accident case."""
    
    return prompt


def load_expected_answer(path):
    """Load the expected answer from text file"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def analyze_constat_few_shot(test_image_path, pipeline=None, model=None, tokenizer=None):
    """
    Analyze a Constat image using few-shot learning.
    Uses one example (example_constat.png + expected_answer_constat.txt) to guide the model.
    Uses PaddleOCR-VL for better OCR quality.
    """
    # ================== STEP 1: OCR PROCESSING (PaddleOCR-VL) ==================
    # We run OCR first, then clear it from memory to save VRAM for MiniCPM
    
    if pipeline is None:
        pipeline = load_paddleocr_vl()
    
    # Check files exist
    if not EXAMPLE_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Example image not found: {EXAMPLE_IMAGE_PATH}\n"
                                f"Please copy your example Constat image to: {EXAMPLE_IMAGE_PATH}")
    
    if not EXPECTED_ANSWER_PATH.exists():
        raise FileNotFoundError(f"Expected answer not found: {EXPECTED_ANSWER_PATH}")

    print("\n📚 Processing example image OCR...")
    example_image = Image.open(EXAMPLE_IMAGE_PATH).convert('RGB')
    example_ocr_texts = extract_ocr_text_vl(pipeline, EXAMPLE_IMAGE_PATH)
    example_prompt = build_training_prompt(example_ocr_texts)
    expected_answer = load_expected_answer(EXPECTED_ANSWER_PATH)
    
    print("\n🎯 Processing test image OCR...")
    test_image = Image.open(test_image_path).convert('RGB')
    test_ocr_texts = extract_ocr_text_vl(pipeline, test_image_path)
    test_prompt = build_test_prompt(test_ocr_texts)
    
    # FREE GPU MEMORY: Delete OCR-VL and clear cache
    print("🧹 Clearing OCR from memory...")
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ================== STEP 2: VLM INFERENCE (MiniCPM-V-2_6) ==================
    
    if model is None or tokenizer is None:
        model, tokenizer = load_minicpm()
    
    print("\n🤖 Running few-shot inference...")
    
    msgs = [
        # Example (one-shot)
        {'role': 'user', 'content': [example_image, example_prompt]},
        {'role': 'assistant', 'content': [expected_answer]},
        # Test
        {'role': 'user', 'content': [test_image, test_prompt]}
    ]
    
    try:
        answer = model.chat(
            image=None,  # Images are in msgs
            msgs=msgs,
            tokenizer=tokenizer
        )
    except Exception as e:
        import traceback
        answer = f"Error during inference: {e}\n\nTraceback:\n{traceback.format_exc()}"
    
    return {
        "test_image": str(test_image_path),
        "example_ocr_texts": example_ocr_texts,
        "test_ocr_texts": test_ocr_texts,
        "analysis": answer
    }


def main():
    """CLI entry point"""
    if len(sys.argv) < 2:
        print("\n❌ Usage: python app_constat_fewshot.py <test_image> [--lang=fr]")
        print("\nThis script uses few-shot learning to analyze French Constats.")
        print("\nRequired files (same directory as this script):")
        print(f"  - {EXAMPLE_IMAGE_PATH.name} : Example Constat image")
        print(f"  - {EXPECTED_ANSWER_PATH.name} : Expected structured answer")
        print("\nExample:")
        print("   python app_constat_fewshot.py new_constat.jpg")
        sys.exit(1)
    
    test_image_path = sys.argv[1]
    
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        sys.exit(1)
    
    # Run analysis
    result = analyze_constat_few_shot(test_image_path)
    
    # Display results
    print("\n" + "=" * 70)
    print("📊 RÉSULTATS DE L'ANALYSE")
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
    
    # Create subdirectory for example
    example_output_dir = output_dir / "example_constat"
    example_output_dir.mkdir(exist_ok=True)
    
    # Save analysis result
    output_path = test_output_dir / "analysis.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result['analysis'])
    print(f"\n💾 Analyse sauvegardée: output/{test_image_name}/analysis.txt")
    
    # Save Example OCR output
    example_ocr_path = example_output_dir / "ocr_text.txt"
    with open(example_ocr_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(result['example_ocr_texts']))
    print(f"💾 Example OCR sauvegardé: output/example_constat/ocr_text.txt")
    
    # Save Test OCR output
    ocr_output_path = test_output_dir / "ocr_text.txt"
    with open(ocr_output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(result['test_ocr_texts']))
    print(f"💾 OCR sauvegardé: output/{test_image_name}/ocr_text.txt")
    
    print("✅ Terminé!")


if __name__ == "__main__":
    main()
