import PyPDF2
import docx
import os # Import os for path handling

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        text = "" # Return empty string on error
    return text

def extract_text_from_docx(docx_path):
    text = []
    try:
        doc = docx.Document(docx_path)
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
    except Exception as e:
        print(f"Error reading DOCX {docx_path}: {e}")
        text = [""] # Return empty list on error
    return "\n".join(text)

def extract_text(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Only PDF and DOCX are supported.")

if __name__ == "__main__":
    print("--- Testing text_extractor.py ---")
    pass
