"""
Simple test script for MiniCPM-V (PaddleOCR VL model).
Usage: python utils/test_minicpm.py <image_path>
"""

import sys
import torch
from pathlib import Path
from PIL import Image

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import os

load_dotenv()

# ================== CONFIG ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path(__file__).parent.parent / "model_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.getenv('HF_TOKEN')


def load_model():
    """Load MiniCPM-V-2_6 model"""
    from transformers import AutoModel, AutoTokenizer
    
    model_name = 'openbmb/MiniCPM-V-2_6'
    
    print(f"Loading {model_name}...")
    
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
    
    print(f"✅ {model_name} loaded on {DEVICE}")
    return model, tokenizer


def run_inference(model, tokenizer, image_path, prompt):
    """Run inference on an image with a prompt"""
    image = Image.open(image_path).convert('RGB')
    
    msgs = [
        {'role': 'user', 'content': [image, prompt]}
    ]
    
    response = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )
    
    return response


def main():
    if len(sys.argv) < 2:
        print("Usage: python utils/test_minicpm.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: {image_path} not found")
        sys.exit(1)
    
    # Default prompt - modify as needed
    prompt = """Describe what you see in this image in detail."""
    
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print("-" * 50)
    
    model, tokenizer = load_model()
    response = run_inference(model, tokenizer, image_path, prompt)
    
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
