#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-Step Qwen3-VL Analysis WITHOUT OCR
Step 1: Analyze Section 12 crop → Extract checkbox states
Step 2: Use checkbox data → Generate full interpretation
"""

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
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

def load_qwen():
    """Load Qwen3-VL-8B-Instruct model"""
    print("📦 Loading Qwen3-VL-8B-Instruct...")
    model_name = 'Qwen/Qwen3-VL-8B-Instruct'
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=str(CACHE_DIR),
        token=HF_TOKEN
    )
    
    # Load model with flash attention if available
    if DEVICE == "cuda":
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                cache_dir=str(CACHE_DIR),
                token=HF_TOKEN
            )
            print("✅ Using Flash Attention 2")
        except:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=str(CACHE_DIR),
                token=HF_TOKEN
            )
            print("⚠️  Flash Attention not available, using default")
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map="auto",
            cache_dir=str(CACHE_DIR),
            token=HF_TOKEN
        )
    
    model.eval()
    print("✅ Qwen3-VL-8B loaded")
    return model, processor


def generate_response(model, processor, messages):
    """Generate response using Qwen3-VL"""
    # Apply chat template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # Generate with VL-optimized parameters
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0
    )
    
    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0] if output_text else ""


# ================== STEP 1: CHECKBOX EXTRACTION ==================

def build_checkbox_prompt():
    """Prompt for Step 1: Extract checkboxes from crop"""
    return """Analyze this Section (Circonstances) checkboxes.

HOW TO DETECT CHECKED BOXES:
- CHECKED: Box contains X, ✓, tick mark, cross, scribble, or any ink/marking inside
- EMPTY: Box is blank/white with no marks inside

For VEHICLE A (Left column) - List all 17 boxes:
Box 1 (stationnement): ☐ EMPTY or ☑ CHECKED (confidence %)
...
Box 17 (signal priorité): ☐ EMPTY or ☑ CHECKED (confidence %)
Same for vehicle B
For VEHICLE B (Right column) - List all 17 boxes:
Box 1 (stationnement): ☐ EMPTY or ☑ CHECKED (confidence %)
...
Box 17 (signal priorité): ☐ EMPTY or ☑ CHECKED (confidence %)
Then
MANUAL COUNTS :
Vehicle A count: [read number]
Vehicle B count: [read number]

VERIFICATION: Your detected count must match the printed count. If mismatch, recheck uncertain boxes, the important part is stating each box what it represents and wether its checked or not.
"""


def extract_checkboxes(model, processor, crop_image_path):
    """Step 1: Analyze crop to extract checkbox states"""
    print("\n📋 STEP 1: Analyzing Section 12 crop for checkboxes...")
    
    prompt = build_checkbox_prompt()
    
    # Build Qwen3-VL message format
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(crop_image_path)},
            {"type": "text", "text": prompt}
        ]
    }]
    
    try:
        checkbox_data = generate_response(model, processor, messages)
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


def generate_full_analysis(model, processor, full_image_path, checkbox_data):
    """Step 2: Generate full interpretation using checkbox results"""
    print("\n📊 STEP 2: Generating full interpretation...")
    
    prompt = build_full_prompt(checkbox_data)
    
    # Build Qwen3-VL message format
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(full_image_path)},
            {"type": "text", "text": prompt}
        ]
    }]
    
    try:
        analysis = generate_response(model, processor, messages)
        print("✅ Full analysis complete")
        return analysis
    except Exception as e:
        import traceback
        error_msg = f"❌ Error in full analysis: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


# ================== MAIN ==================

def test_two_step_analysis(test_image_path, output_dir):
    """Two-step VLM analysis without OCR"""
    print("\n🧪 TWO-STEP QWEN3-VL ANALYSIS (NO OCR)\n")
    print("=" * 70)
    
    test_image_path = Path(test_image_path)
    
    # Load model once
    model, processor = load_qwen()
    
    # Step 0: Crop Section 12 and save to output directory
    print("\n✂️  STEP 0: Cropping Section 12...")
    crop_path = extract_section_12_crop(str(test_image_path), output_dir=output_dir)
    if not crop_path:
        print("❌ Cropping failed")
        return None
    
    # Step 1: Extract checkboxes from crop
    checkbox_data = extract_checkboxes(model, processor, crop_path)
    
    # Save immediately after Step 1
    with open(output_dir / "checkboxes.txt", 'w', encoding='utf-8') as f:
        f.write(checkbox_data)
    print(f"💾 Saved: {output_dir}/checkboxes.txt")
    
    # Step 2: Generate full analysis using checkbox data
    full_analysis = generate_full_analysis(model, processor, str(test_image_path), checkbox_data)
    
    return {
        'checkbox_data': checkbox_data,
        'full_analysis': full_analysis,
        'crop_path': crop_path
    }


def main():
    """CLI entry point"""
    if len(sys.argv) < 2:
        print("\n❌ Usage: python test_qwen_twostep.py <test_image>")
        print("\nExample:")
        print("   python test_qwen_twostep.py images/4.jpg")
        sys.exit(1)
    
    test_image_path = sys.argv[1]
    
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        sys.exit(1)
    
    # Create output directory structure: output_qwen/imagename/
    test_name = Path(test_image_path).stem
    output_dir = Path(__file__).parent / "output_qwen" / test_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run two-step analysis
    results = test_two_step_analysis(test_image_path, output_dir)
    
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
    
    # Save Step 2 results (Step 1 already saved)
    with open(output_dir / "analysis.txt", 'w', encoding='utf-8') as f:
        f.write(results['full_analysis'])
    
    print(f"\n💾 All results saved in: output_qwen/{test_name}/")
    print(f"   - {Path(results['crop_path']).name}")
    print(f"   - checkboxes.txt (saved after Step 1)")
    print(f"   - analysis.txt")
    print("\n✅ Two-step analysis complete!")


if __name__ == "__main__":
    main()
