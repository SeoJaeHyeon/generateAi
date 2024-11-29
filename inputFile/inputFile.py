import base64
from PyPDF2 import PdfReader

def encode_base64_content_from_file(file_path: str) -> str:
    """Encode file content to base64 format."""
    with open(file_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode("utf-8")
    return encoded_string

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def preview_file(file):
    """Preview the uploaded file (PDF or Image)."""
    if file is None:
        return None, None  # 파일이 없으면 미리보기도 없도록 반환
    
    if file.name.endswith((".jpg", ".jpeg", ".png")):
        return file.name, None  # Return image preview  
    elif file.name.endswith(".pdf"):
        with open(file.name, "rb") as pdf_file:
            base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
        pdf_preview = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500px"></iframe>'
        return None, pdf_preview  # Return PDF preview