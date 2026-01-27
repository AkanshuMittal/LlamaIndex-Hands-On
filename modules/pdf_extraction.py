import pdfplumber

def extract_profile_from_pdf(uploaded_file) -> str:
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    if not text.strip():
        raise ValueError("No text found in PDF")

    return text