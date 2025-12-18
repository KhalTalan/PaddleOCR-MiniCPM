import streamlit as st
import os
import sys
import shutil
from pathlib import Path
from PIL import Image
import torch
import gc
import subprocess
import json
from dotenv import load_dotenv

# Add current directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent))

# Import specific project utils
try:
    from utils.crop_utils import extract_section_12_crop
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
except ImportError as e:
    st.error(f"Failed to import project modules: {e}")
    st.stop()

# Load environment variables
load_dotenv()

# Constants
REAL_ESRGAN_DIR = Path("Real_ESRGAN")
OUTPUT_DIR = Path("output")
CROPS_DIR = OUTPUT_DIR / "crops"
GAN_DIR = OUTPUT_DIR / "gan"
CACHE_DIR = Path.home() / ".cache" / "dual_vlm"

# Ensure directories exist
CROPS_DIR.mkdir(parents=True, exist_ok=True)
GAN_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
st.set_page_config(
    page_title="Constat Amiable Analyzer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
        margin-top: 2rem;
    }
    .card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        color: #155724;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions (Adapted from test_qwen_twostep.py) ---

def run_real_esrgan_in_process():
    """Run Real-ESRGAN in-process to share CUDA context"""
    try:
        from omegaconf import OmegaConf
        # Add internal path for imports to work
        if str(REAL_ESRGAN_DIR) not in sys.path:
            sys.path.append(str(REAL_ESRGAN_DIR))
            
        from real_esrgan.apis.super_resolution import SuperResolutionInferencer
        
        config_path = REAL_ESRGAN_DIR / "configs" / "inference" / "images.yaml"
        if not config_path.exists():
            return False, f"Config not found: {config_path}"
            
        config_dict = OmegaConf.load(config_path)
        

        # Instantiate and run - FORCE CPU TO AVOID OOM
        config_dict.DEVICE = "cpu"
        # Explicitly set paths relative to the current working directory or absolute paths
        config_dict.INPUTS = str(CROPS_DIR)
        config_dict.OUTPUT = str(GAN_DIR)
        
        inferencer = SuperResolutionInferencer(config_dict)
        # inferencer.warmup() # Skip warmup to save memory
        inferencer.inference()
        
        # Cleanup
        del inferencer
        clean_memory()
        
        return True, "Success"
    except Exception as e:
        import traceback
        return False, f"{e}\n{traceback.format_exc()}"

def load_qwen_model():
    """Load Qwen3-VL-8B-Instruct model (uncached to manage memory)"""
    model_name = 'Qwen/Qwen3-VL-8B-Instruct'
    hf_token = os.getenv('HF_TOKEN')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=str(CACHE_DIR),
        token=hf_token
    )
    
    # Load model
    if device == "cuda":
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                cache_dir=str(CACHE_DIR),
                token=hf_token
            )
        except:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=str(CACHE_DIR),
                token=hf_token
            )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map="auto",
            cache_dir=str(CACHE_DIR),
            token=hf_token
        )
    
    model.eval()
    return model, processor

def generate_llm_response(model, processor, messages):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0
    )
    
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

def build_checkbox_prompt():
    return """
TASK: Analyze Section 12 "CIRCONSTANCES" checkboxes on the provided image.

RULES:
- **Important dont confuse the last box (18th) as a marqued checkbox, its for the total number of marqued checkboxes**.
- Each label applies to both Vehicle A (left column) and Vehicle B (right column).
- CHECKED = any ink/mark inside the box (X, ✓, cross, scribble, dot, partial stroke).
- EMPTY = completely blank/white inside box.
- UNCERTAIN = smudge, low contrast, cropped, or ambiguous marks (provide justification).
- MISSING = box is missing from the crop (provide justification).
- Ignore ink outside the box unless >50% falls inside the box interior.
- Provide integer confidence 0-100; explain if confidence <70%.
- **Important dont confuse the last box (18th) as a marqued checkbox, its for the total number of marqued checkboxes**.

OUTPUT FORMAT (JSON ONLY):
{
  "boxes": [
    {
      "index": 1,
      "label": "stationnement",
      "vehicle_A": {"status":"CHECKED|EMPTY|UNCERTAIN|MISSING","confidence":0-100,"justification":""},
      "vehicle_B": {"status":"CHECKED|EMPTY|UNCERTAIN|MISSING","confidence":0-100,"justification":""}
    },
    ...
    {
      "index": 17,
      "label": "signal de priorité",
      "vehicle_A": {"status":"CHECKED|EMPTY|UNCERTAIN|MISSING","confidence":0-100,"justification":""},
      "vehicle_B": {"status":"CHECKED|EMPTY|UNCERTAIN|MISSING","confidence":0-100,"justification":""}
    }
  ],
  "marked_cases": {
    "vehicle_A":{"value":0,"confidence":0-100,"justification":""},
    "vehicle_B":{"value":0,"confidence":0-100,"justification":""}
  },
  "meta":{"image_id":"<optional>","timestamp":"<iso>","notes":["anything unusual like skew, crop, bleed"]}
}

IMPORTANT: Output ONLY the JSON, no extra text.
"""

def build_full_prompt(checkbox_data):
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


# --- Application Layout ---


def clean_memory():
    """Aggressively clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def main():
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Project Overview", "Analyze Report"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This tool uses Qwen3-VL and Real-ESRGAN to analyze French 'Constat Amiable' accident reports."
    )

    if page == "Project Overview":
        render_overview()
    elif page == "Analyze Report":
        render_analysis()

def render_overview():
    st.markdown('<div class="main-header">🚗 Constat Amiable Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Project Explanation
    
    This project automates the analysis of French automobile accident reports (Constat Amiable) using a **two-step Vision-Language Model (VLM)** pipeline with super-resolution enhancement.
    
    #### 🔄 The Workflow
    
    The system follows a specific pipeline to ensure accuracy:
    
    1. **Input Image**: You upload the photo of the accident report.
    2. **PaddleOCR Detection & Cropping**: The system automatically detects and crops "Section 12 - Circonstances". This is the most critical part where checkboxes determine fault.
    3. **Real-ESRGAN Enhancement**: The cropped section is upscaled (Super Resolution) to make small handwritten crosses and checkboxes clear.
    4. **Step 1: Checkbox Extraction**: Qwen3-VL analyzes the *enhanced* crop to accurately read the checkboxes (Vehicle A vs B).
    5. **Step 2: Full Analysis**: The model looks at the *full original image* + the *verified checkbox data* to generate a comprehensive legal and fault analysis.
    
    #### 🧠 Technologies Used
    - **Qwen3-VL-8B**: State-of-the-art vision-language model.
    - **Real-ESRGAN**: GAN for image super-resolution.
    - **PaddleOCR**: Optical Character Recognition for precise cropping.
    """)
    
    st.image("https://mermaid.ink/img/pako:eNptkctqwzAQRX9FzKpA_QAfpJAu2k03XbTQy2I9tSxbylFSCPn3yk5ioKULWc259848RjOohIxoR_W24sXyAOcVzTucF2x5X_F6sWBLfv-W84ItHyt2XiwY5wXlW863fFyw9WPFbouF4Lygec-Lgo2PBTsvFpLzgnZe8L7l44KdHyt2WywU5wXtWz5WbH1fcfBiITkv6N7zomD7x4K9FwvFeUH_vpcFWz9W7L1YKM8Lune8Ktj-sWDvxUJFaUml1VzQ43o4nU9XQy6Xw_VyPDwM-Xw-na_Hw9OQ6-V0vR4PT0O-Xs7X6_HwNOT75Xy_Hg9PQ35czvfH8fA05Mf1fH8cDx_2Pwx53B_Hjw_DkKf9cfz8MAx5PhzHzw_DkJfDcfz6MAx5PRzHr6vlMJb_h_H7Tz-M5T-O33_5YSz_cfz-yw9j-Y_j919-GMt_HL__8sNY_uP4_ZcfxvIfx--__DCW_zh-_20fhvF_o_8A43CK1g?type=png", caption="System Workflow")

def render_analysis():
    st.markdown('<div class="main-header">📊 Analyze Accident Report</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Constat Image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Save temp file
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        img_path = temp_dir / uploaded_file.name
        
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Determine the target crop filename to check for GAN result later
        # Logic from test_qwen_twostep.py:
        # crop_path = extract_section_12_crop(test_image_path) -> saves to output/crops/filename_crop_section12.jpg
        # enhanced_crop_path = GAN_DIR / crop_filename
        crop_filename_expected = f"{img_path.stem}_crop_section12.jpg"
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(str(img_path), caption="Original Image", use_container_width=True)
        

        if st.button("🚀 Start Analysis", type="primary"):
            # Clean memory immediately
            clean_memory()
            
            status_container = st.empty()
            
            # placeholders for cleanup
            model = None
            processor = None
            
            try:
                # --- Step 0: Crop ---
                status_container.info("✂️ Step 1/4: Cropping Section 12 (Circonstances)...")
                crop_path = extract_section_12_crop(str(img_path), output_dir=CROPS_DIR)
                
                if not crop_path:
                    st.error("Failed to crop the image. Please try a clearer image.")
                    return
                
                # Show crop
                with col2:
                    st.image(crop_path, caption="Detected Crop", use_container_width=True)
                
                # --- Step 0.5: GAN ---
                status_container.info("🎨 Step 2/4: Enhancing Image with Real-ESRGAN...")
                
                # Check if enhanced exists
                enhanced_path = GAN_DIR / Path(crop_path).name
                
                # We just run it.
                success, msg = run_real_esrgan_in_process()
                
                if success and enhanced_path.exists():
                    analysis_crop = str(enhanced_path)
                    st.success("GAN Enhancement Successful!")
                    # Show GAN result
                    with col2:
                        st.image(analysis_crop, caption="Enhanced (GAN) Crop", use_container_width=True)
                else:
                    st.warning(f"GAN Enhancement failed or skipped ({msg}). Using original crop.")
                    analysis_crop = crop_path
                
                # --- Clear Memory before Model Load ---
                status_container.info("pwipe Checking GPU memory...")
                # Double check memory again before loading heavy model
                clean_memory()
                
                # --- Step 1 & 2: Qwen Analysis ---
                status_container.info("📦 Step 3/4: Loading Qwen3-VL Model (This may take a moment)...")
                
                model, processor = load_qwen_model()
                
                # Extraction
                status_container.info("📋 Step 4/4: Extracting Checkboxes...")
                
                # Checkbox Prompts
                prompt_checkbox = build_checkbox_prompt()
                messages_checkbox = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": analysis_crop},
                        {"type": "text", "text": prompt_checkbox}
                    ]
                }]
                
                checkbox_data = generate_llm_response(model, processor, messages_checkbox)
                
                # Show intermediate JSON
                with st.expander("View Checkbox Data (JSON)"):
                    st.code(checkbox_data, language="json")
                
                # Full Analysis
                status_container.info("📊 Generating Final Report...")
                
                prompt_full = build_full_prompt(checkbox_data)
                messages_full = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img_path)},
                        {"type": "text", "text": prompt_full}
                    ]
                }]
                
                full_analysis = generate_llm_response(model, processor, messages_full)
                
                # --- Final Output ---
                status_container.success("Analysis Complete! ✅")
                
                st.markdown("### 📄 Final Analysis Report")
                st.markdown(full_analysis)
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            
            finally:
                # Force cleanup of model and processor
                if model is not None:
                    del model
                if processor is not None:
                    del processor
                
                # Aggressive memory cleanup
                clean_memory()

if __name__ == "__main__":
    main()
