# 🚗 French Constat Analysis with PaddleOCR + MiniCPM-V

Automated analysis of French automobile accident reports (Constat Amiable d'Accident Automobile) using **Few-Shot Learning** with vision-language models.

## 🎯 Overview

This project combines:
- **PaddleOCR** for French text extraction
- **MiniCPM-V-2_6** for intelligent document understanding
- **Few-shot learning** to teach the model without extensive training data

### What It Does

Given a French Constat Amiable image, the system:
1. ✅ Extracts all accident details (date, time, location, parties)
2. ✅ Identifies which circumstance boxes are checked in Section 12
3. ✅ Analyzes driver observations and identifies blame statements
4. ✅ Reconstructs the accident step-by-step
5. ✅ Determines fault liability based on French traffic law
6. ✅ Generates structured analysis with percentage liability recommendations

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended)
- Hugging Face account with access to MiniCPM-V-2_6

### Installation

```bash
# Clone the repository
git clone https://github.com/KhalTalan/PaddleOCR-MiniCPM.git
cd PaddleOCR-MiniCPM

# Install dependencies
pip install torch torchvision paddleocr transformers pillow python-dotenv

# For CUDA support (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Setup

1. **Get Hugging Face Token**
   - Visit https://huggingface.co/settings/tokens
   - Create a token
   - Accept terms at https://huggingface.co/openbmb/MiniCPM-V-2_6

2. **Create `.env` file**
   ```bash
   # Copy the example
   cp .env.example .env
   
   # Edit and add your token
   HF_TOKEN=your_huggingface_token_here
   ```

3. **Verify Setup**
   ```bash
   python app_constat_fewshot.py --help
   ```

---

## 📖 Usage

### Basic Usage

```bash
python app_constat_fewshot.py path/to/constat_image.jpg
```

### Example

```bash
python app_constat_fewshot.py images/3.png
```

### Output Files

All outputs are saved to the `output/` directory:

| File | Content |
|------|---------|
| `{filename}_constat_result.txt` | Complete structured analysis |
| `{filename}_ocr_output.txt` | Raw OCR extracted text |
| `example_constat_ocr.txt` | OCR from training example (for debugging) |

---

## 🧠 How It Works: Few-Shot Learning

### The Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  TRAINING EXAMPLE (One-Shot)                                    │
├─────────────────────────────────────────────────────────────────┤
│  example_constat.png  ──┐                                       │
│                         ├──> PaddleOCR ──> OCR Text ──> Prompt  │
│  expected_answer.txt ───┘                                  │    │
│                                                            ▼    │
│                                                    ┌──────────┐ │
│                                                    │ MiniCPM  │ │
│  NEW CONSTAT (Test)                               │    V     │ │
├─────────────────────────────────────────────────  │  2_6     │ │
│  test_image.png  ───────> PaddleOCR ──> OCR Text ──> Prompt │ │
│                                                    └──────────┘ │
│                                                            │    │
│                                                            ▼    │
│                                                   Generated     │
│                                                   Analysis      │
└─────────────────────────────────────────────────────────────────┘
```

### Conversation Structure

The model receives a 3-turn conversation:

```python
[
    # Turn 1: User shows example
    {
        'role': 'user',
        'content': [example_image, prompt_with_instructions]
    },
    
    # Turn 2: Assistant shows perfect response
    {
        'role': 'assistant', 
        'content': [expected_answer]
    },
    
    # Turn 3: User asks to analyze new case
    {
        'role': 'user',
        'content': [test_image, prompt_with_instructions]
    }
    # Model generates analysis here ↓
]
```

The model learns the pattern from the example and applies it to new cases!

---

## 📊 Output Format

The analysis follows a 7-section structure:

### 1. ACCIDENT DETAILS
- Date, time, location
- Injuries status
- Witness information

### 2. VEHICLE A (Left Side)
- Driver information
- Vehicle details
- Insurance information
- Damage description
- Driver observation (with blame analysis)

### 3. VEHICLE B (Right Side)
- Same structure as Vehicle A

### 4. CIRCUMSTANCES (Section 12)
- Lists ONLY checked boxes for each vehicle
- Example: `Vehicle A: Box 8 CHECKED (rear-end collision)`

### 5. ACCIDENT RECONSTRUCTION
- Step-by-step explanation
- Evidence citations (box numbers, damage patterns)

### 6. FAULT ANALYSIS
- Applies French Barème de Responsabilité rules
- Assigns liability percentages
- Provides reasoning based on circumstances

### 7. SUMMARY
- Brief conclusion with fault determination

---

## 🔧 Configuration

### Model Settings

Edit `app_constat_fewshot.py` to configure:

```python
# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model parameters (in load_minicpm function)
attn_implementation='sdpa'  # Options: 'sdpa', 'flash_attention_2'
torch_dtype=torch.bfloat16  # Options: bfloat16, float16, float32
```

### OCR Language

```python
# Default is French
ocr = load_paddleocr(lang='fr')

# For multilingual support, change to:
ocr = load_paddleocr(lang='latin')  # Latin script languages
```

---

## 📁 Project Structure

```
PaddleOCR-MiniCPM/
├── app_constat_fewshot.py      # Main analysis script
├── expected_answer_constat.txt # Training example answer
├── example_constat.png         # Training example image
├── .env.example                # Environment template
├── .env                        # Your HF token (git-ignored)
├── README.md                   # This file
├── output/                     # Generated analyses
│   ├── 3_constat_result.txt
│   ├── 3_ocr_output.txt
│   └── example_constat_ocr.txt
└── images/                     # Test images
    └── 3.png
```

---

## 🎓 Understanding the Code

### Key Functions

| Function | Purpose |
|----------|---------|
| `load_paddleocr()` | Initialize French OCR engine |
| `load_minicpm()` | Load MiniCPM-V-2_6 with authentication |
| `extract_ocr_text()` | Extract text blocks from image |
| `build_prompt()` | Create analysis prompt with OCR text |
| `analyze_constat_few_shot()` | Main few-shot learning pipeline |

### Critical Design Decisions

1. **Split Prompts (Training vs Test)**: Uses a detailed prompt for the example to teach the format, but a constrained prompt for the test case to prevent data hallucination.
2. **OCR + Vision**: Combines text extraction with visual understanding (for checkboxes)
3. **Concise expected answer**: ~2.3KB to avoid token limits
4. **Source citations**: Every fact traceable to document section
5. **Blame detection**: Identifies when driver observations accuse the other party

---

## 🐛 Troubleshooting

### Authentication Errors

```
401 Client Error: Unauthorized
```

**Solution**: 
1. Check your `.env` file has the correct `HF_TOKEN`
2. Verify you accepted the model terms at https://huggingface.co/openbmb/MiniCPM-V-2_6

### Out of Memory

```
CUDA out of memory
```

**Solution**:
- Reduce image resolution before processing
- Use `torch_dtype=torch.float16` instead of `bfloat16`
- Use CPU mode (slower): `DEVICE = "cpu"`

### Incomplete Output

If the model cuts off mid-analysis:
- The expected answer might be too long
- Try reducing `expected_answer_constat.txt` further
- Check token limits in model settings

---

## 🔬 Advanced Usage

### Custom Training Example

To use your own training example:

1. Replace `example_constat.png` with your image
2. Update `expected_answer_constat.txt` with the correct analysis
3. Follow the existing format (7 sections)
4. Keep it concise (~2.3KB max)

### Batch Processing

```python
import glob
from pathlib import Path

# Process all images in a directory
for img_path in glob.glob("images/*.png"):
    print(f"\nProcessing {img_path}...")
    os.system(f"python app_constat_fewshot.py {img_path}")
```

---

## 📝 Example Output

```
CONSTAT AMIABLE ANALYSIS

1. ACCIDENT DETAILS
Date: 09/10/2024, Time: 12h41
Location: Rue de la Libération, 42000 Saint-Étienne, France
Injuries: No | Other damage: No

2. VEHICLE A (Left/Blue)
Driver: FAURE Aymerick, DOB: 18/10/2000
Vehicle: Renault Clio 3, Reg: 722-FXL-92
Damage: Front bumper damaged
Observation: "N'avait pas de clignotant!" - BLAMES Vehicle B

3. VEHICLE B (Right/Yellow)
Driver: KERVEAN Anne, DOB: 30/04/1995
Vehicle: Peugeot 206, Reg: DG-789-TK
Damage: Front right fender and bumper
Observation: "Était sur son téléphone!" - BLAMES Vehicle A

4. CIRCUMSTANCES
Vehicle A: Box 8 CHECKED (rear-end collision)
Vehicle B: Box 12 CHECKED (turning right)

6. FAULT ANALYSIS
Vehicle A: 75-100% liability (rear-end collision)
Vehicle B: 0-25% liability (possible failure to signal)
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Support for more Constat layouts
- Multi-language support
- Automated validation against ground truth
- Integration with insurance systems

---

## 📄 License

This project is open source. Check the repository for license details.

---

## 🙏 Acknowledgments

- **PaddleOCR** for excellent OCR capabilities
- **OpenBMB** for the MiniCPM-V-2_6 model
- **Hugging Face** for model hosting

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

**Repository**: https://github.com/KhalTalan/PaddleOCR-MiniCPM