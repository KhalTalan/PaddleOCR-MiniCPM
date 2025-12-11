"""
Simple test script for Qwen3-VL model.
Usage: python utils/test_qwen.py <image_path>
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


def load_model():
    """Load Qwen3-VL model"""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    cache_dir = Path(__file__).parent.parent / "model_cache"
    hf_token = os.getenv("HF_TOKEN")
    
    print(f"Loading {model_name}...")
    
    try:
        # Try with flash attention
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            cache_dir=str(cache_dir),
            token=hf_token
        )
        print("   Using Flash Attention 2")
    except:
        # Fallback to SDPA
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
            cache_dir=str(cache_dir),
            token=hf_token
        )
        print("   Using SDPA attention")
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        token=hf_token
    )
    
    return model, processor


def run_inference(model, processor, image_path, prompt):
    """Run inference on an image with a prompt"""
    from qwen_vl_utils import process_vision_info
    
    image = Image.open(image_path).convert("RGB")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False
        )
    
    output_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    return response


def main():
    if len(sys.argv) < 2:
        print("Usage: python utils/test_qwen.py <image_path>")
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
    
    model, processor = load_model()
    response = run_inference(model, processor, image_path, prompt)
    
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
