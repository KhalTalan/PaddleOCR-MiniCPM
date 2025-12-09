#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test MiniCPM-V 2-Shot WITHOUT OCR
Uses two examples to test VLM's visual checkbox detection capability
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

# Paths - now using examples folder
EXAMPLES_DIR = Path(__file__).parent / "examples"
EXAMPLE1_IMAGE = EXAMPLES_DIR / "example_constat1.png"
EXAMPLE1_ANSWER = EXAMPLES_DIR / "expected_answer1.txt"
EXAMPLE2_IMAGE = EXAMPLES_DIR / "example_constat2.jpg"
EXAMPLE2_ANSWER = EXAMPLES_DIR / "expected_answer2.txt"

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
    """Simple direct prompt"""
    return """Analyze this French Constat Amiable image.

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


def load_text(path):
    """Load text file"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def test_minicpm_2shot(test_image_path):
    """Test MiniCPM-V with 2-shot, NO OCR"""
    print("\n🧪 Testing MiniCPM-V 2-Shot Learning (NO OCR)\n")
    
    # Check files
    for f in [EXAMPLE1_IMAGE, EXAMPLE1_ANSWER, EXAMPLE2_IMAGE, EXAMPLE2_ANSWER]:
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f}")
    
    # Load model
    model, tokenizer = load_minicpm()
    
    # Load images
    print("\n📸 Loading images...")
    example1_img = Image.open(EXAMPLE1_IMAGE).convert('RGB')
    example1_ans = load_text(EXAMPLE1_ANSWER)
    example2_img = Image.open(EXAMPLE2_IMAGE).convert('RGB')
    example2_ans = load_text(EXAMPLE2_ANSWER)
    test_img = Image.open(test_image_path).convert('RGB')
    
    prompt = build_prompt()
    
    print("\n🤖 Running 2-shot inference...\n")
    print(f"   Example 1: {EXAMPLE1_IMAGE.name}")
    print(f"   Example 2: {EXAMPLE2_IMAGE.name}")
    print(f"   Test: {Path(test_image_path).name}\n")
    
    # 2-shot messages
    msgs = [
        # Example 1
        {'role': 'user', 'content': [example1_img, prompt]},
        {'role': 'assistant', 'content': [example1_ans]},
        # Example 2
        {'role': 'user', 'content': [example2_img, prompt]},
        {'role': 'assistant', 'content': [example2_ans]},
        # Test
        {'role': 'user', 'content': [test_img, "Analyze this NEW Constat image using the same format:"]}
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
        print("   python test_minicpm_only.py images/4.jpg")
        sys.exit(1)
    
    test_image_path = sys.argv[1]
    
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        sys.exit(1)
    
    # Run test
    result = test_minicpm_2shot(test_image_path)
    
    # Display
    print("=" * 70)
    print("📊 MINICPM-V 2-SHOT RESULT (NO OCR)")
    print("=" * 70)
    print(result)
    print("=" * 70)
    
    # Save
    output_dir = Path(__file__).parent / "output_minicpm_test"
    output_dir.mkdir(exist_ok=True)
    
    test_name = Path(test_image_path).stem
    output_path = output_dir / f"{test_name}_result.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"\n💾 Result saved: output_minicpm_test/{test_name}_result.txt")
    print("✅ Test complete!")


if __name__ == "__main__":
    main()

