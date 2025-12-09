#!/usr/bin/env python3
"""
Fixed Installation Script for PaddleOCR + MiniCPM-V with Flash Attention
Properly installs flash-attn for CUDA 11.8 + PyTorch 2.1.2
"""

import subprocess
import sys
import time

def run_command(cmd, description, ignore_errors=False):
    """Execute command and display result"""
    print(f"\n{'='*70}")
    print(f"📦 {description}")
    print(f"{'='*70}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} - OK")
        return True
    except subprocess.CalledProcessError as e:
        if ignore_errors:
            print(f"⚠️  {description} - Error ignored")
            return True
        else:
            print(f"❌ {description} - FAILED")
            return False

def check_installation():
    """Verify installation"""
    print(f"\n{'='*70}")
    print("🔍 INSTALLATION VERIFICATION")
    print(f"{'='*70}\n")
    
    checks = {
        "PyTorch": ("import torch; print(f'Version: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')", 30),
        "Flash Attention": ("import flash_attn; print(f'Version: {flash_attn.__version__}')", 30),
        "PaddlePaddle": ("import paddle; print(f'Version: {paddle.__version__}')", 30),
        "PaddleOCR": ("from paddleocr import PaddleOCR; print('Import OK')", 30),
        "PaddleOCR-VL": ("from paddleocr import PaddleOCRVL; print('VL Support OK')", 30),
        "Transformers": ("import transformers; print(f'Version: {transformers.__version__}')", 10),
    }
    
    results = []
    for name, (code, timeout) in checks.items():
        try:
            print(f"Checking {name}... ", end='', flush=True)
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode == 0:
                print(f"✅ {result.stdout.strip()}")
                results.append(True)
            else:
                print(f"❌ Error: {result.stderr.strip()[:100]}")
                results.append(False)
        except subprocess.TimeoutExpired:
            print(f"⏱️  Timeout ({timeout}s)")
            results.append(False)
        except Exception as e:
            print(f"❌ Exception: {str(e)[:100]}")
            results.append(False)
    
    return all(results)

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║        FIXED Installation: PaddleOCR + MiniCPM-V + Flash Attention  ║
║        Lightning Studio Compatible                                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}\n")
    
    installations = []
    
    # STEP 1: Core PyTorch (compatible with flash-attn)
    installations.extend([
        ("pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118", 
         "PyTorch 2.1.2 + CUDA 11.8", False),
    ])
    
    # STEP 2: Install Flash Attention FIRST (before transformers)
    # Using prebuilt wheel for CUDA 11.8 + PyTorch 2.1 + Python 3.11
    installations.extend([
        ("pip install ninja packaging wheel", 
         "Build dependencies for flash-attn", False),
        
        # Install specific prebuilt wheel from a reliable source
        # For CUDA 11.8 + PyTorch 2.1 + Python 3.11 (Linux x86_64)
        ("pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl", 
         "Flash Attention 2.5.8 (prebuilt wheel for CUDA 11.8)", False),
    ])
    
    # STEP 3: Fix NumPy (CRITICAL for PaddleOCR)
    installations.append(
        ("pip install 'numpy<2.0' --force-reinstall",
         "NumPy <2.0 (required by PaddleOCR)", False)
    )
    
    # STEP 4: PaddlePaddle 3.2.1+ (REQUIRED for PaddleOCR-VL)
    installations.append(
        ("pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/",
         "PaddlePaddle GPU 3.2.1 (required for PaddleOCR-VL)", False)
    )
    
    # STEP 5: Install safetensors special version (REQUIRED for PaddleOCR-VL)
    # Different for Linux vs Windows
    installations.append(
        ("pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl",
         "SafeTensors (PaddleOCR-VL requirement)", True)  # Ignore errors if on Windows
    )
    
    # STEP 6: PaddleOCR with VL support
    installations.extend([
        ("pip install opencv-python opencv-python-headless",
         "OpenCV", False),
        
        ("pip install shapely pyclipper lmdb imgaug",
         "Geometric dependencies", False),
        
        ("pip install -U \"paddleocr[doc-parser]\"",
         "PaddleOCR with VL (document parser) support", False),
        
        ("pip install tqdm rapidfuzz beautifulsoup4 lxml premailer openpyxl",
         "PaddleOCR additional dependencies", True),
    ])
    
    # STEP 7: Transformers and related (AFTER flash-attn)
    installations.extend([
        ("pip install transformers==4.44.2",
         "Transformers 4.44.2", False),
        
        ("pip install accelerate==0.34.2",
         "Accelerate 0.34.2", False),
        
        ("pip install sentencepiece protobuf",
         "SentencePiece and Protobuf", False),
        
        ("pip install timm==1.0.9",
         "TIMM (PyTorch Image Models)", False),
    ])
    
    # STEP 7: Utilities
    installations.append(
        ("pip install Pillow scipy scikit-image pyyaml requests",
         "General utilities", True)
    )
    
    # Install sequentially
    print("\n🚀 Starting installation...\n")
    print("⏱️  This may take 10-15 minutes (flash-attn compilation is slow)\n")
    
    success_count = 0
    total_count = len(installations)
    
    for cmd, desc, ignore_errors in installations:
        if run_command(cmd, desc, ignore_errors):
            success_count += 1
        time.sleep(1)
    
    # Verification
    print("\n")
    print(f"📊 Progress: {success_count}/{total_count} steps completed\n")
    
    verification_success = check_installation()
    
    # Summary
    print(f"\n{'='*70}")
    if verification_success:
        print("✅ INSTALLATION COMPLETE!")
        print(f"{'='*70}")
        print("\n🎉 All components operational including Flash Attention!\n")
        print("🚀 Usage:")
        print("   python app_dual_vlm.py your_image.png")
    else:
        print("⚠️  PARTIAL INSTALLATION")
        print(f"{'='*70}")
        print("\nMost components installed.")
        print("Check errors above, especially flash-attn compilation.\n")
        print("💡 If flash-attn failed, try:")
        print("   pip install flash-attn==2.5.8 --no-build-isolation")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()