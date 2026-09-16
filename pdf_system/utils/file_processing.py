import tempfile
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance

def extract_text_from_pdf(pdf_path: str) -> str:
    """Enhanced PDF text extraction with multiple fallback strategies"""
    try:
        # Try PyPDF2 first
        text = ""
        with open(pdf_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # If minimal text extracted, try OCR
        if len(text.strip().split()) < 100:  # Threshold for considering OCR
            try:
                images = convert_from_path(
                    pdf_path,
                    dpi=300,
                    grayscale=True,
                    thread_count=4
                )
                ocr_text = ""
                for img in images:
                    img = img.convert('L')
                    img = ImageEnhance.Contrast(img).enhance(2.0)
                    ocr_text += pytesseract.image_to_string(img) + "\n"
                
                if len(ocr_text.strip().split()) > len(text.strip().split()):
                    text = ocr_text
            except Exception as ocr_error:
                raise Warning(f"OCR processing had issues: {str(ocr_error)}")
        
        return text if text.strip() else None
    except Exception as e:
        raise Exception(f"PDF processing failed: {str(e)}")

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary location and return path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.getbuffer())
        return temp_pdf.name