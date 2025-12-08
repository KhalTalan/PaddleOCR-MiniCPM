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

# ================== CONFIG ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path.home() / ".cache" / "dual_vlm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(CACHE_DIR / "hf")
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR / "transformers")

# Paths to few-shot example
EXAMPLE_IMAGE_PATH = Path(__file__).parent / "example_constat.png"
EXPECTED_ANSWER_PATH = Path(__file__).parent / "expected_answer_constat.txt"

print(f"🚀 Device: {DEVICE}")

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
        cache_dir=str(CACHE_DIR)
    )
    
    if DEVICE == "cuda":
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation='sdpa',  # or 'flash_attention_2' if available
            torch_dtype=torch.bfloat16,
            cache_dir=str(CACHE_DIR)
        )
        model = model.eval().cuda()
    else:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.float32,
            cache_dir=str(CACHE_DIR)
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


def build_prompt_with_ocr(ocr_texts):
    """
    Build the analysis prompt combining OCR results with extraction instructions.
    This prompt guides the model on how to interpret the Constat form.
    """
    ocr_content = "\n".join(ocr_texts)
    
    prompt = f"""You are an expert insurance analyst specializing in French automobile accident reports (Constat Amiable d'Accident Automobile).

OCR EXTRACTED TEXT:
{ocr_content}

TASK: Analyze this French Constat Amiable and provide a complete analysis. For each piece of information, cite the source section. Use simple plain text format.

OUTPUT STRUCTURE:

1. ACCIDENT DETAILS
Extract: Date, time, location, injuries (yes/no), property damage, witness info.
Always cite the source section in parentheses.

2. VEHICLE A (Left/Yellow Side)
Extract in subsections:
- POLICYHOLDER (Section 6): name, address, phone
- VEHICLE (Section 7): type, make/model, registration, country
- INSURANCE (Section 8): company, contract number, green card, validity, agency
- DRIVER (Section 9): name, DOB, address, phone, license details
- CIRCUMSTANCES CHECKED (Section 12): list only the box numbers that are checked
- IMPACT POINT (Section 10A): describe where the arrow points
- VISIBLE DAMAGE (Section 11A): quote exactly what is written
- DRIVER OBSERVATION (Section 14A): quote exactly what is written, then add critical analysis stating whether this is a self-description or a BLAME statement against the other driver

3. VEHICLE B (Right/Blue Side)
Same structure as Vehicle A.

4. ACCIDENT SKETCH (Section 13)
Describe what is visible: street names, vehicle positions, signatures present.

5. ACCIDENT RECONSTRUCTION
Step by step, explain how the accident happened based only on:
- Which circumstance boxes are checked
- The impact points shown
- The damage described
Cite the evidence for each step.

6. FAULT DETERMINATION
Apply French Constat liability rules:
- Box 1 (parked) = 0% liability
- Box 2 (leaving parking) = 100% liability
State the percentage for each vehicle with reasoning.

7. SUMMARY
Brief summary with date, location, what happened, and fault conclusion.

IMPORTANT: Only state facts visible in the document. If something is unclear, write "Not legible". Never guess."""
    
    return prompt


def load_expected_answer(path):
    """Load the expected answer from text file"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def analyze_constat_few_shot(test_image_path, ocr=None, model=None, tokenizer=None):
    """
    Analyze a Constat image using few-shot learning.
    Uses one example (example_constat.png + expected_answer_constat.txt) to guide the model.
    """
    # Load models if not provided
    if ocr is None:
        ocr = load_paddleocr()
    if model is None or tokenizer is None:
        model, tokenizer = load_minicpm()
    
    # ================== PREPARE FEW-SHOT EXAMPLE ==================
    print("\n📚 Preparing few-shot example...")
    
    # Load example image and expected answer
    if not EXAMPLE_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Example image not found: {EXAMPLE_IMAGE_PATH}\n"
                                f"Please copy your example Constat image to: {EXAMPLE_IMAGE_PATH}")
    
    if not EXPECTED_ANSWER_PATH.exists():
        raise FileNotFoundError(f"Expected answer not found: {EXPECTED_ANSWER_PATH}")
    
    example_image = Image.open(EXAMPLE_IMAGE_PATH).convert('RGB')
    example_ocr_texts = extract_ocr_text(ocr, EXAMPLE_IMAGE_PATH)
    example_prompt = build_prompt_with_ocr(example_ocr_texts)
    expected_answer = load_expected_answer(EXPECTED_ANSWER_PATH)
    
    # ================== PREPARE TEST IMAGE ==================
    print("\n🎯 Preparing test image...")
    
    test_image = Image.open(test_image_path).convert('RGB')
    test_ocr_texts = extract_ocr_text(ocr, test_image_path)
    test_prompt = build_prompt_with_ocr(test_ocr_texts)
    
    # ================== BUILD FEW-SHOT MESSAGES ==================
    print("\n🤖 Running few-shot inference...")
    
    msgs = [
        # Example (one-shot)
        {'role': 'user', 'content': [example_image, example_prompt]},
        {'role': 'assistant', 'content': [expected_answer]},
        # Test
        {'role': 'user', 'content': [test_image, test_prompt]}
    ]
    
    # ================== INFERENCE ==================
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
        "ocr_texts": test_ocr_texts,
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
    
    print(f"\n📝 Texte extrait ({len(result['ocr_texts'])} blocs):")
    for i, text in enumerate(result['ocr_texts'][:5], 1):
        preview = text[:80] + "..." if len(text) > 80 else text
        print(f"   {i}. {preview}")
    
    if len(result['ocr_texts']) > 5:
        print(f"   ... et {len(result['ocr_texts']) - 5} autres")
    
    print("\n🤖 Analyse structurée:")
    print("-" * 70)
    print(result['analysis'])
    print("-" * 70)
    
    # Save result to output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / (Path(test_image_path).stem + "_constat_result.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result['analysis'])
    print(f"\n💾 Résultat sauvegardé: {output_path}")
    print("✅ Terminé!")


if __name__ == "__main__":
    main()
