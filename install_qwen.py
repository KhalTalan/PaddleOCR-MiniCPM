#!/usr/bin/env python3
"""
Install script for Qwen3-VL-8B-Instruct with Flash Attention 2
Run: python install_qwen.py
"""

import subprocess
import sys
import platform

def run(cmd, check=True):
    print(f"\n🔧 {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode == 0

def get_flash_attn_wheel_url():
    """Get the correct prebuilt wheel URL from GitHub releases"""
    # Detect Python version
    py_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    
    # Detect CUDA version from torch
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda.replace(".", "")[:3]  # "12.1" -> "121"
        else:
            cuda_version = "121"  # Default to CUDA 12.1
    except:
        cuda_version = "121"
    
    # Flash Attention 2.8.3 wheel URL pattern
    version = "2.8.3"
    base_url = f"https://github.com/Dao-AILab/flash-attention/releases/download/v{version}"
    
    # Linux wheel
    if platform.system() == "Linux":
        wheel_name = f"flash_attn-{version}+cu{cuda_version}-{py_version}-{py_version}-linux_x86_64.whl"
    else:
        # Windows/Mac - no prebuilt wheels, will try pip
        return None
    
    return f"{base_url}/{wheel_name}"

def main():
    print("=" * 60)
    print("🚀 Qwen3-VL-8B Installation with Flash Attention 2")
    print("=" * 60)
    
    # 1. Upgrade pip
    run(f"{sys.executable} -m pip install --upgrade pip")
    
    # 2. Install PyTorch with CUDA
    print("\n📦 Installing PyTorch with CUDA support...")
    run(f"{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    
    # 3. Install transformers (latest for Qwen3-VL support)
    print("\n📦 Installing transformers (>=4.57.0 for Qwen3-VL)...")
    run(f"{sys.executable} -m pip install 'transformers>=4.57.0'")
    
    # 4. Install Flash Attention 2 from prebuilt wheel
    print("\n⚡ Installing Flash Attention 2...")
    wheel_url = get_flash_attn_wheel_url()
    
    if wheel_url:
        print(f"   Downloading: {wheel_url}")
        if not run(f"{sys.executable} -m pip install {wheel_url}", check=False):
            print("   ⚠️  Prebuilt wheel failed, trying pip install...")
            run(f"{sys.executable} -m pip install flash-attn --no-build-isolation", check=False)
    else:
        print("   No prebuilt wheel for this platform, trying pip...")
        run(f"{sys.executable} -m pip install ninja packaging")
        run(f"{sys.executable} -m pip install flash-attn --no-build-isolation", check=False)
    
    # 5. Install other dependencies
    print("\n📦 Installing additional dependencies...")
    run(f"{sys.executable} -m pip install accelerate pillow python-dotenv opencv-python")
    
    # 6. Optional: qwen-vl-utils for image processing
    print("\n📦 Installing qwen-vl-utils...")
    run(f"{sys.executable} -m pip install qwen-vl-utils", check=False)
    
    # Verify installation
    print("\n" + "=" * 60)
    print("✅ Verifying installation...")
    print("=" * 60)
    
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"   ❌ PyTorch: {e}")
    
    try:
        import transformers
        print(f"   Transformers: {transformers.__version__}")
    except Exception as e:
        print(f"   ❌ Transformers: {e}")
    
    try:
        import flash_attn
        print(f"   ✅ Flash Attention: {flash_attn.__version__}")
    except Exception as e:
        print(f"   ⚠️  Flash Attention: Not installed ({e})")
        print("      (Model will still work, just slower)")
    
    print("\n" + "=" * 60)
    print("🎉 Installation complete!")
    print("=" * 60)
    print("\nTest with:")
    print("   python test_qwen_twostep.py images/4.jpg")

if __name__ == "__main__":
    main()
