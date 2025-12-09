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

def load_paddleocr(lang='fr'):
    """Load PaddleOCR for French text extraction"""
    print("📦 Loading PaddleOCR...")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang=lang,
        use_gpu=(DEVICE == "cuda"),
        show_log=False
    )
    return ocr


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


def extract_ocr_text(ocr, image_path):
    """Extract text using PaddleOCR"""
    print(f"🔍 OCR: {Path(image_path).name}")
    result = ocr.ocr(str(image_path), cls=True)
    texts = []
    
    if result and result[0]:
        for line in result[0]:
            if len(line) >= 2 and line[1]:
                text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                if text:
                    texts.append(text.strip())
    
    print(f"   Found {len(texts)} text blocks")
    return texts


def build_training_prompt(ocr_texts):
    """
    Detailed prompt for the training example.
    Teaches the model the format using the example data.
    """
    ocr_content = "\n".join(ocr_texts)
    
    prompt = f"""Analyze this French Constat Amiable (accident report) and provide a structured analysis.

OCR TEXT:
{ocr_content}

Follow this 7-section format:
1. Accident details (date, time, location, injuries, witness)
2. Vehicle A - Extract all info + damage + driver observation
3. Vehicle B - Extract all info + damage + driver observation
4. Circumstances - CRITICAL: Carefully check Section 12 image, only list CHECKED boxes
5. Reconstruction - Step-by-step based on checked boxes + damage
6. Fault analysis - Apply French liability rules, assign percentages with reasoning
7. Summary - Brief conclusion

KEY REMINDERS:
- Driver observations: Quote exactly, then state if it's a BLAME against the other driver
- Only state facts visible in the document
- Write "Not legible" if unclear
- Be concise and accurate"""
    
    return prompt


def build_test_prompt(ocr_texts):
    """
    Prompt for the NEW test image.
    Includes strong constraints to prevent bleeding from the example.
    """
    ocr_content = "\n".join(ocr_texts)
    
    prompt = f"""Analyze this NEW French Constat Amiable.

CRITICAL INSTRUCTION:
This is a COMPLETELY DIFFERENT accident case from the previous example.
- IGNORE all names, dates, and details from the previous example.
- Use ONLY the information visible in the NEW image and the NEW OCR text below.
- Do NOT hallucinate information from the previous turn.

NEW OCR TEXT:
{ocr_content}

Provide the analysis in the same 7-section format as the example, but using ONLY the data from this new accident case."""
    
    return prompt


def load_expected_answer(path):
    """Load the expected answer from text file"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


import gc

def analyze_constat_few_shot(test_image_path, ocr=None, model=None, tokenizer=None):
    """
    Analyze a Constat image using few-shot learning.
    Uses one example (example_constat.png + expected_answer_constat.txt) to guide the model.
    """
    # ================== STEP 1: OCR PROCESSING (PaddleOCR) ==================
    # We run OCR first, then clear it from memory to save VRAM for MiniCPM
    
    if ocr is None:
        ocr = load_paddleocr()
    
    # Check files exist
    if not EXAMPLE_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Example image not found: {EXAMPLE_IMAGE_PATH}\n"
                                f"Please copy your example Constat image to: {EXAMPLE_IMAGE_PATH}")
    
    if not EXPECTED_ANSWER_PATH.exists():
        raise FileNotFoundError(f"Expected answer not found: {EXPECTED_ANSWER_PATH}")

    print("\n📚 Processing example image OCR...")
    example_image = Image.open(EXAMPLE_IMAGE_PATH).convert('RGB')
    example_ocr_texts = extract_ocr_text(ocr, EXAMPLE_IMAGE_PATH)
    example_prompt = build_training_prompt(example_ocr_texts)
    expected_answer = load_expected_answer(EXPECTED_ANSWER_PATH)
    
    print("\n🎯 Processing test image OCR...")
    test_image = Image.open(test_image_path).convert('RGB')
    test_ocr_texts = extract_ocr_text(ocr, test_image_path)
    test_prompt = build_test_prompt(test_ocr_texts)
    
    # FREE GPU MEMORY: Delete OCR and clear cache
    print("🧹 Clearing OCR from memory...")
    del ocr
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
    
    
    # Save results to output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save analysis result
    output_path = output_dir / (Path(test_image_path).stem + "_constat_result.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result['analysis'])
    print(f"\n💾 Analyse sauvegardée: {output_path}")
    
    # Save Example OCR output
    example_ocr_path = output_dir / "example_constat_ocr.txt"
    with open(example_ocr_path, 'w', encoding='utf-8') as f:
        f.write("EXAMPLE OCR EXTRACTED TEXT\n")
        f.write("=" * 70 + "\n\n")
        for i, text in enumerate(result['example_ocr_texts'], 1):
            f.write(f"{i}. {text}\n")
    print(f"💾 Example OCR sauvegardé: {example_ocr_path}")
    
    # Save Test OCR output
    test_ocr_path = output_dir / (Path(test_image_path).stem + "_ocr_output.txt")
    with open(test_ocr_path, 'w', encoding='utf-8') as f:
        f.write("TEST OCR EXTRACTED TEXT\n")
        f.write("=" * 70 + "\n\n")
        for i, text in enumerate(result['test_ocr_texts'], 1):
            f.write(f"{i}. {text}\n")
    print(f"💾 Test OCR sauvegardé: {test_ocr_path}")
    
    print("✅ Terminé!")


if __name__ == "__main__":
    main()
