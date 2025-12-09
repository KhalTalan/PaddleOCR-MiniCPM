#!/usr/bin/env python3
"""
Complete Installation Script for Qwen3-VL-8B-Instruct
Installs everything from scratch: PyTorch, Transformers, Flash Attention 2

Run: python install_qwen.py
"""

import subprocess
import sys
import platform

def run(cmd, check=True):
    print(f"\n🔧 {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode == 0

def main():
    print("=" * 60)
    print("🚀 Qwen3-VL-8B Complete Installation")
    print("=" * 60)
    print(f"   Platform: {platform.system()}")
    print(f"   Python: {sys.version.split()[0]}")
    
    # 1. Upgrade pip
    run(f"{sys.executable} -m pip install --upgrade pip")
    
    # 2. Install PyTorch with CUDA 12.4 (latest stable)
    print("\n📦 Step 1: Installing PyTorch 2.5.1 with CUDA 12.4...")
    run(f"{sys.executable} -m pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124")
    
    # 3. Install transformers
    print("\n📦 Step 2: Installing Transformers >=4.57.0...")
    run(f"{sys.executable} -m pip install 'transformers>=4.57.0'")
    
    # 4. Install Flash Attention 2
    print("\n⚡ Step 3: Installing Flash Attention 2...")
    # Install build dependencies first
    run(f"{sys.executable} -m pip install ninja packaging wheel", check=False)
    # Try precompiled wheel
    if not run(f"{sys.executable} -m pip install flash-attn --no-build-isolation", check=False):
        print("   ⚠️  Flash Attention build failed - will use SDPA instead")
    
    # 5. Install other dependencies
    print("\n📦 Step 4: Installing supporting packages...")
    run(f"{sys.executable} -m pip install accelerate pillow python-dotenv opencv-python einops")
    
    # 6. Install qwen-vl-utils
    print("\n📦 Step 5: Installing qwen-vl-utils...")
    run(f"{sys.executable} -m pip install qwen-vl-utils", check=False)
    
    # Verify installation
    print("\n" + "=" * 60)
    print("✅ Verifying Installation")
    print("=" * 60)
    
    errors = []
    
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"      CUDA: {torch.version.cuda}")
            print(f"      GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("      ⚠️ CUDA not available - CPU only")
    except Exception as e:
        errors.append(f"PyTorch: {e}")
        print(f"   ❌ PyTorch: {e}")
    
    try:
        import transformers
        print(f"   ✅ Transformers: {transformers.__version__}")
    except Exception as e:
        errors.append(f"Transformers: {e}")
        print(f"   ❌ Transformers: {e}")
    
    try:
        from transformers import Qwen3VLForConditionalGeneration
        print(f"   ✅ Qwen3-VL model class available")
    except Exception as e:
        errors.append(f"Qwen3-VL: {e}")
        print(f"   ❌ Qwen3-VL: {e}")
    
    try:
        import flash_attn
        print(f"   ✅ Flash Attention: {flash_attn.__version__}")
    except:
        print(f"   ⚠️ Flash Attention: Not installed (SDPA will be used)")
    
    print("\n" + "=" * 60)
    if errors:
        print("❌ Installation has errors:")
        for e in errors:
            print(f"   - {e}")
    else:
        print("🎉 Installation complete!")
    print("=" * 60)
    print("\nTest with:")
    print("   python test_qwen_twostep.py images/4.jpg")

if __name__ == "__main__":
    main()
