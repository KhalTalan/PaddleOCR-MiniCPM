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

def get_flash_attn_wheel():
    """Get correct prebuilt wheel URL for the environment"""
    # Detect Python version
    py_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    
    # Flash Attention v2.6.3 wheels available for torch 2.4, cu123
    # Using torch 2.4 compatible wheels with PyTorch 2.5 (should work)
    # Format: flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-{py}-{py}-linux_x86_64.whl
    
    base_url = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3"
    wheel_name = f"flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-{py_version}-{py_version}-linux_x86_64.whl"
    
    return f"{base_url}/{wheel_name}"

def main():
    print("=" * 60)
    print("🚀 Qwen3-VL-8B Complete Installation")
    print("=" * 60)
    print(f"   Platform: {platform.system()}")
    print(f"   Python: {sys.version.split()[0]}")
    
    # 1. Upgrade pip
    run(f"{sys.executable} -m pip install --upgrade pip")
    
    # 2. Fix NumPy version (CRITICAL - scipy/sklearn need numpy<2.0)
    print("\n📦 Step 1: Fixing NumPy version (downgrade to <2.0)...")
    run(f"{sys.executable} -m pip install 'numpy<2.0'")
    
    # 3. Install PyTorch with CUDA 12.4
    print("\n📦 Step 2: Installing PyTorch 2.5.1 with CUDA 12.4...")
    run(f"{sys.executable} -m pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124")
    
    # 4. Install transformers
    print("\n📦 Step 3: Installing Transformers >=4.57.0...")
    run(f"{sys.executable} -m pip install 'transformers>=4.57.0'")
    
    # 4. Install Flash Attention 2 from prebuilt wheel
    print("\n⚡ Step 3: Installing Flash Attention 2 (prebuilt wheel)...")
    # Uninstall old version first
    run(f"{sys.executable} -m pip uninstall flash-attn -y", check=False)
    
    # Install prebuilt wheel from GitHub releases
    wheel_url = get_flash_attn_wheel()
    print(f"   Downloading: {wheel_url}")
    if not run(f"{sys.executable} -m pip install {wheel_url}", check=False):
        print("   ⚠️  Prebuilt wheel failed, trying pip build...")
        run(f"{sys.executable} -m pip install ninja packaging")
        run(f"{sys.executable} -m pip install flash-attn --no-build-isolation", check=False)
    
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
