#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-Step MiniCPM-V Analysis WITHOUT OCR
Step 1: Analyze Section 12 crop → Extract checkbox states
Step 2: Use checkbox data → Generate full interpretation
"""

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import os
import sys
from dotenv import load_dotenv

# Import cropping utility
sys.path.insert(0, str(Path(__file__).parent))
from utils.crop_utils import extract_section_12_crop

load_dotenv()

# ================== CONFIG ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path.home() / ".cache" / "dual_vlm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(CACHE_DIR / "hf")
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR / "transformers")

HF_TOKEN = os.getenv('HF_TOKEN')

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


# ================== STEP 1: CHECKBOX EXTRACTION ==================

def build_checkbox_prompt():
    """Prompt for Step 1: Extract checkboxes from crop"""
    return """Analyze this Section 12 (Circonstances) checkbox image.

For VEHICLE A (Left column) - List all 17 boxes:
Box 1 (stationnement): ☐ EMPTY or ☑ CHECKED (confidence %)
Box 2 (quittait stationnement): ☐ EMPTY or ☑ CHECKED (confidence %)
...
Box 17 (signal priorité): ☐ EMPTY or ☑ CHECKED (confidence %)

For VEHICLE B (Right column) - List all 17 boxes:
Box 1 (stationnement): ☐ EMPTY or ☑ CHECKED (confidence %)
...
Box 17 (signal priorité): ☐ EMPTY or ☑ CHECKED (confidence %)

Summary: Vehicle A checked boxes: [list], Vehicle B checked boxes: [list]"""


def extract_checkboxes(model, tokenizer, crop_image_path):
    """Step 1: Analyze crop to extract checkbox states"""
    print("\n📋 STEP 1: Analyzing Section 12 crop for checkboxes...")
    
    crop_img = Image.open(crop_image_path).convert('RGB')
    prompt = build_checkbox_prompt()
    
    msgs = [
        {'role': 'user', 'content': [crop_img, prompt]}
    ]
    
    try:
        checkbox_data = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer
        )
        print("✅ Checkbox extraction complete")
        return checkbox_data
    except Exception as e:
        import traceback
        error_msg = f"❌ Error in checkbox extraction: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


# ================== STEP 2: FULL INTERPRETATION ==================

def build_full_prompt(checkbox_data):
    """Prompt for Step 2: Generate full interpretation using checkbox data"""
    return f"""Analyze this French Constat Amiable accident report.

I have already verified the Section 12 checkboxes. Here are the results:

{checkbox_data}

Using this checkbox information, provide a complete analysis with the following sections:

1. ACCIDENT DETAILS
Date, Time, Location, Injuries, Other damage, Witnesses

2. VEHICLE A (Left)
Driver name, Address, Vehicle details, Insurance, Damage description, Driver's observation

3. VEHICLE B (Right)
Driver name, Address, Vehicle details, Insurance, Damage description, Driver's observation

4. CIRCUMSTANCES SUMMARY
Based on the verified checkboxes above, summarize what each vehicle was doing.

5. ACCIDENT RECONSTRUCTION
Step-by-step description of what happened based on the checked boxes.

6. FAULT ANALYSIS
Who is likely at fault and why, based on French traffic law and the circumstances.

7. CONCLUSION
Brief summary (2-3 sentences) of the accident and fault determination."""


def generate_full_analysis(model, tokenizer, full_image_path, checkbox_data):
    """Step 2: Generate full interpretation using checkbox results"""
    print("\n📊 STEP 2: Generating full interpretation...")
    
    full_img = Image.open(full_image_path).convert('RGB')
    prompt = build_full_prompt(checkbox_data)
    
    msgs = [
        {'role': 'user', 'content': [full_img, prompt]}
    ]
    
    try:
        analysis = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer
        )
        print("✅ Full analysis complete")
        return analysis
    except Exception as e:
        import traceback
        error_msg = f"❌ Error in full analysis: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


# ================== MAIN ==================

def test_two_step_analysis(test_image_path):
    """Two-step VLM analysis without OCR"""
    print("\n🧪 TWO-STEP VLM ANALYSIS (NO OCR)\n")
    print("=" * 70)
    
    test_image_path = Path(test_image_path)
    
    # Load model once
    model, tokenizer = load_minicpm()
    
    # Step 0: Crop Section 12
    print("\n✂️  STEP 0: Cropping Section 12...")
    crop_path = extract_section_12_crop(str(test_image_path))
    if not crop_path:
        print("❌ Cropping failed")
        return None
    
    # Step 1: Extract checkboxes from crop
    checkbox_data = extract_checkboxes(model, tokenizer, crop_path)
    
    # Step 2: Generate full analysis using checkbox data
    full_analysis = generate_full_analysis(model, tokenizer, str(test_image_path), checkbox_data)
    
    return {
        'checkbox_data': checkbox_data,
        'full_analysis': full_analysis
    }


def main():
    """CLI entry point"""
    if len(sys.argv) < 2:
        print("\n❌ Usage: python test_minicpm_twostep.py <test_image>")
        print("\nExample:")
        print("   python test_minicpm_twostep.py images/4.jpg")
        sys.exit(1)
    
    test_image_path = sys.argv[1]
    
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        sys.exit(1)
    
    # Run two-step analysis
    results = test_two_step_analysis(test_image_path)
    
    if not results:
        sys.exit(1)
    
    # Display results
    print("\n" + "=" * 70)
    print("📋 STEP 1 RESULT: CHECKBOX EXTRACTION")
    print("=" * 70)
    print(results['checkbox_data'])
    
    print("\n" + "=" * 70)
    print("📊 STEP 2 RESULT: FULL ANALYSIS")
    print("=" * 70)
    print(results['full_analysis'])
    print("=" * 70)
    
    # Save results
    output_dir = Path(__file__).parent / "output_twostep"
    output_dir.mkdir(exist_ok=True)
    
    test_name = Path(test_image_path).stem
    
    # Save checkpoint data
    with open(output_dir / f"{test_name}_checkboxes.txt", 'w', encoding='utf-8') as f:
        f.write(results['checkbox_data'])
    
    # Save full analysis
    with open(output_dir / f"{test_name}_analysis.txt", 'w', encoding='utf-8') as f:
        f.write(results['full_analysis'])
    
    print(f"\n💾 Results saved:")
    print(f"   - output_twostep/{test_name}_checkboxes.txt")
    print(f"   - output_twostep/{test_name}_analysis.txt")
    print("\n✅ Two-step analysis complete!")


if __name__ == "__main__":
    main()
