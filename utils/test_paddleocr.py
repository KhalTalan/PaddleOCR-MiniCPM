"""
Simple test script for PaddleOCR-VL (Vision-Language OCR).
Usage: python utils/test_paddleocr.py <image_path>
"""

import sys
from pathlib import Path


def load_paddleocr_vl():
    """Load PaddleOCR-VL pipeline"""
    from paddleocr import PaddleOCRVL
    
    print("Loading PaddleOCR-VL...")
    
    pipeline = PaddleOCRVL(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    
    print("✅ PaddleOCR-VL loaded")
    return pipeline


def run_ocr(pipeline, image_path, output_dir=None):
    """Run OCR on an image"""
    image_path = Path(image_path)
    
    if output_dir is None:
        output_dir = image_path.parent / f"{image_path.stem}_ocr_output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing: {image_path}")
    
    # Run pipeline
    result = pipeline.predict(str(image_path))
    
    # Save outputs
    for res in result:
        res.save_to_json(save_path=str(output_dir / "paddleocr_vl.json"))
        res.save_to_markdown(save_path=str(output_dir / "paddleocr_vl.md"))
    
    # Read markdown output
    markdown_file = output_dir / "paddleocr_vl.md"
    if markdown_file.exists():
        with open(markdown_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    return "No output generated"


def main():
    if len(sys.argv) < 2:
        print("Usage: python utils/test_paddleocr.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: {image_path} not found")
        sys.exit(1)
    
    print(f"Image: {image_path}")
    print("-" * 50)
    
    pipeline = load_paddleocr_vl()
    result = run_ocr(pipeline, image_path)
    
    print("\nOCR Result (Markdown):")
    print(result)


if __name__ == "__main__":
    main()
