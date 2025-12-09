#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test MiniCPM-V Few-Shot WITHOUT OCR
Direct image-to-analysis to test VLM's visual checkbox detection capability
"""

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ================== CONFIG ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path.home() / ".cache" / "dual_vlm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(CACHE_DIR / "hf")
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR / "transformers")

HF_TOKEN = os.getenv('HF_TOKEN')

# Paths
EXAMPLE_IMAGE_PATH = Path(__file__).parent / "example_constat.png"
EXPECTED_ANSWER_PATH = Path(__file__).parent / "expected_answer_constat.txt"

print(f"🚀 Device: {DEVICE}")
if HF_TOKEN:
    print(f"🔑 HF Token loaded: {HF_TOKEN[:10]}...")

# ================== MODEL ==================

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


def build_prompt():
    """Simple direct prompt without OCR"""
    prompt = """Analyze this French accident report (Constat Amiable) image.

Provide structured output in 7 sections:

1. ACCIDENT DETAILS: Date, Time, Location, Injuries, Other damage, Witnesses

2. VEHICLE A (Left): Driver, Address, Vehicle, Insurance, Damage, Observation

3. VEHICLE B (Right): Same as Vehicle A

4. CIRCUMSTANCES (Section 12) - CHECKBOX VERIFICATION:
List all 17 boxes for Vehicle A: Each marked ☐ EMPTY or ☑ CHECKED with confidence %
List all 17 boxes for Vehicle B: Each marked ☐ EMPTY or ☑ CHECKED with confidence %
Provide summary of checked boxes

5. RECONSTRUCTION: What happened based on checked boxes

6. FAULT ANALYSIS: Who is at fault and why

7. SUMMARY: Brief conclusion"""
    
    return prompt


def load_expected_answer(path):
    """Load expected answer"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def test_minicpm_fewshot(test_image_path):
    """Test MiniCPM-V with few-shot, NO OCR"""
    print("\n🧪 Testing MiniCPM-V Few-Shot (NO OCR)\n")
    
    # Check files
    if not EXAMPLE_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Example image not found: {EXAMPLE_IMAGE_PATH}")
    if not EXPECTED_ANSWER_PATH.exists():
        raise FileNotFoundError(f"Expected answer not found: {EXPECTED_ANSWER_PATH}")
    
    # Load model
    model, tokenizer = load_minicpm()
    
    # Load images
    print("\n📸 Loading images...")
    example_image = Image.open(EXAMPLE_IMAGE_PATH).convert('RGB')
    test_image = Image.open(test_image_path).convert('RGB')
    
    # Load expected answer
    expected_answer = load_expected_answer(EXPECTED_ANSWER_PATH)
    
    # Build prompt
    prompt = build_prompt()
    
    print("\n🤖 Running few-shot inference (direct images, no OCR)...\n")
    
    # Few-shot messages
    msgs = [
        # Example
        {'role': 'user', 'content': [example_image, prompt]},
        {'role': 'assistant', 'content': [expected_answer]},
        # Test
        {'role': 'user', 'content': [test_image, "Analyze this NEW Constat image using the same format:"]}
    ]
    
    try:
        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer
        )
    except Exception as e:
        import traceback
        answer = f"Error: {e}\n\n{traceback.format_exc()}"
    
    return answer


def main():
    """CLI entry point"""
    if len(sys.argv) < 2:
        print("\n❌ Usage: python test_minicpm_only.py <test_image>")
        print("\nExample:")
        print("   python test_minicpm_only.py images/Constat-Accident-velo.jpg")
        sys.exit(1)
    
    test_image_path = sys.argv[1]
    
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        sys.exit(1)
    
    # Run test
    result = test_minicpm_fewshot(test_image_path)
    
    # Display
    print("=" * 70)
    print("📊 MINICPM-V FEW-SHOT RESULT (NO OCR)")
    print("=" * 70)
    print(result)
    print("=" * 70)
    
    # Save
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    test_name = Path(test_image_path).stem
    output_path = output_dir / f"{test_name}_minicpm_only.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"\n💾 Result saved: {output_path}")
    print("✅ Test complete!")


if __name__ == "__main__":
    main()
