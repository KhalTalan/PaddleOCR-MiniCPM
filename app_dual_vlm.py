#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document Analysis with PaddleOCR + MiniCPM-o-2_6 (Fixed)
"""

import torch
from PIL import Image
import json
import sys
from pathlib import Path
import os

# ================== CONFIG ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path.home() / ".cache" / "dual_vlm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set cache paths
os.environ['HF_HOME'] = str(CACHE_DIR / "hf")
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR / "transformers")

print(f"🚀 Device: {DEVICE} | Cache: {CACHE_DIR}")

# ================== MAIN CLASS ==================

class DocumentAnalyzer:
    def __init__(self, lang='en'):
        self.lang = lang
        self.ocr = None
        self.vlm = None
        self.tokenizer = None
        
    def load_ocr(self):
        """Load PaddleOCR once"""
        if self.ocr:
            return
        print("📦 Loading PaddleOCR...")
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=self.lang,
            use_gpu=(DEVICE == "cuda"),
            show_log=False
        )
        
    def load_vlm(self):
        """Load MiniCPM-o-2_6 once"""
        if self.vlm:
            return
        print("📦 Loading MiniCPM-o-2_6...")
        from transformers import AutoModel, AutoTokenizer
        
        model_name = "openbmb/MiniCPM-o-2_6"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(CACHE_DIR)
        )
        
        # Use the CORRECT configuration as per the model's requirements
        print("   Loading with flash_attention_2 and bfloat16...")
        
        if DEVICE == "cuda":
            self.vlm = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation='flash_attention_2',  # Model requires this (not eager!)
                torch_dtype=torch.bfloat16,  # Use bfloat16 as recommended
                cache_dir=str(CACHE_DIR)
            )
            self.vlm = self.vlm.eval().cuda()
        else:
            # CPU fallback (slower, but works)
            self.vlm = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation='sdpa',  # Use SDPA for CPU
                torch_dtype=torch.float32,
                cache_dir=str(CACHE_DIR)
            )
            self.vlm = self.vlm.eval()
        
        print("✅ Models loaded")
    
    def extract_text(self, img_path):
        """Extract text with OCR"""
        self.load_ocr()
        print(f"🔍 OCR: {Path(img_path).name}")
        
        result = self.ocr.ocr(str(img_path), cls=True)
        texts = []
        
        if result and result[0]:
            for line in result[0]:
                if len(line) >= 2 and line[1]:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    if text:
                        texts.append(text.strip())
        
        print(f"   Found {len(texts)} text blocks")
        return texts
    
    def analyze(self, img_path, question=None, enable_thinking=False):
        """Full analysis: OCR + VLM"""
        # Get OCR text
        texts = self.extract_text(img_path)
        
        # Load image for VLM
        img = Image.open(img_path).convert('RGB')
        
        # Analyze with VLM
        self.load_vlm()
        print("🤖 VLM Analysis...")
        
        if not question:
            question = "Analyze this document thoroughly. Provide: document type, key information (names, dates, amounts, addresses), structure and layout, and any important observations or details."
        
        # Add OCR context
        if texts:
            ocr_sample = "\n".join(texts[:10])[:800]
            question += f"\n\nOCR extracted text sample:\n{ocr_sample}"
        
        # Use the CORRECT message format: content is a list with [image, question]
        msgs = [{'role': 'user', 'content': [img, question]}]
        
        try:
            # Use the correct chat signature
            response = self.vlm.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                enable_thinking=enable_thinking
            )
        except Exception as e:
            import traceback
            response = f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}"
        
        return {
            "texts": texts,
            "analysis": response,
            "image": str(img_path)
        }
    
    def multi_turn_chat(self, img_path, questions):
        """Multi-turn conversation about an image"""
        texts = self.extract_text(img_path)
        img = Image.open(img_path).convert('RGB')
        
        self.load_vlm()
        
        msgs = []
        responses = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n🤖 Question {i}: {question}")
            
            # First question includes the image
            if i == 1:
                msgs.append({'role': 'user', 'content': [img, question]})
            else:
                msgs.append({'role': 'user', 'content': [question]})
            
            try:
                answer = self.vlm.chat(
                    msgs=msgs,
                    tokenizer=self.tokenizer
                )
                print(f"💬 Answer {i}: {answer}")
                
                # Add assistant response to history
                msgs.append({"role": "assistant", "content": [answer]})
                responses.append({"question": question, "answer": answer})
                
            except Exception as e:
                error_msg = f"Error: {e}"
                print(f"❌ {error_msg}")
                responses.append({"question": question, "answer": error_msg})
        
        return {
            "texts": texts,
            "conversation": responses,
            "image": str(img_path)
        }

# ================== CLI ==================

def main():
    if len(sys.argv) < 2:
        print("\n❌ Usage: python app_dual_vlm.py <image> [question] [--lang=en|ch|fr] [--thinking]")
        print("\nExamples:")
        print("   python app_dual_vlm.py doc.jpg")
        print('   python app_dual_vlm.py invoice.png "What is the total amount?"')
        print('   python app_dual_vlm.py doc.jpg --thinking  # Enable long-thinking mode')
        print('   python app_dual_vlm.py doc.jpg --lang=fr')
        sys.exit(1)
    
    img_path = sys.argv[1]
    question = None
    lang = 'en'
    enable_thinking = False
    
    for arg in sys.argv[2:]:
        if arg.startswith('--lang='):
            lang = arg.split('=')[1]
        elif arg == '--thinking':
            enable_thinking = True
        elif not arg.startswith('--'):
            question = arg
    
    if not Path(img_path).exists():
        print(f"❌ File not found: {img_path}")
        sys.exit(1)
    
    # Analyze
    analyzer = DocumentAnalyzer(lang=lang)
    result = analyzer.analyze(img_path, question, enable_thinking=enable_thinking)
    
    # Display results
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    
    print(f"\n📝 Extracted Text ({len(result['texts'])} blocks):")
    for i, text in enumerate(result['texts'][:5], 1):
        preview = text[:80] + "..." if len(text) > 80 else text
        print(f"   {i}. {preview}")
    
    if len(result['texts']) > 5:
        print(f"   ... and {len(result['texts']) - 5} more")
    
    print("\n🤖 Analysis:")
    print("-" * 70)
    print(result['analysis'])
    print("-" * 70)
    
    # Save
    output = Path(img_path).stem + "_result.json"
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved: {output}")
    print("✅ Done!")

if __name__ == "__main__":
    main()