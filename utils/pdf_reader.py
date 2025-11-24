

import PyPDF2

def read_pdf(file_path):
    """
    Reads a PDF file and returns its text content.
    
    Args:
        file_path (str): Path to the PDF file
    
    Returns:
        str: Text extracted from PDF
    """
    text = ""
    try:
 
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            

            for page in reader.pages:
                text += page.extract_text() + "\n"
                
        return text
    
    except Exception as e:
        return f"Error reading PDF: {e}"
