import pdfplumber
from PIL import Image
import io
 
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text if text.strip() else "No text found"
 
def extract_tables_from_pdf(pdf_path):
    """Extracts tables from a PDF file."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            if extracted_tables:
                tables.extend(extracted_tables)
    return tables if tables else "No tables found"
 
# Example usage
pdf_path = "sample_pdf.pdf"  # Change this to your PDF file
 
text_content = extract_text_from_pdf(pdf_path)
print("\nExtracted Text:\n", text_content)
 
tables_content = extract_tables_from_pdf(pdf_path)
print("\nExtracted Tables:\n", tables_content)
 
