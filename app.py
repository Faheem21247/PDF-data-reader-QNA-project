import os
import uuid
from flask import Flask, render_template, request, jsonify
import pdfplumber
from utils.text_cleaner import clean_text
from utils.summarizer import generate_summary
from utils import qa as qa_utils

# Optional: Set OCR path manually (Windows users)
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_FILE_MB = 20  # Max PDF size


@app.route('/')
def upload_page():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files.get('pdf_file')
    if not file or file.filename == "":
        return render_template('error.html', message="No file selected.")

    # Only PDF allowed
    if not file.filename.lower().endswith('.pdf'):
        return render_template('error.html', message="Only PDF files are allowed.")

    # Check file size
    file.seek(0, os.SEEK_END)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if size_mb > MAX_FILE_MB:
        return render_template('error.html', message=f"PDF too large ({size_mb:.1f} MB). Max allowed is {MAX_FILE_MB} MB.")

    # Save file
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(save_path)

    # Extract text with OCR fallback
    pdf_text = ""
    try:
        with pdfplumber.open(save_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    pdf_text += text + "\n"
                else:
                    try:
                        from PIL import Image
                        import pytesseract
                        img = page.to_image(resolution=300).original
                        ocr_text = pytesseract.image_to_string(img)
                        if ocr_text.strip():
                            pdf_text += ocr_text + "\n"
                    except Exception as ocr_error:
                        print(f"OCR Error (page {page_num}):", ocr_error)
    except Exception as e:
        return render_template("error.html", message=f"Error opening PDF: {str(e)}")

    if not pdf_text.strip():
        return render_template("error.html", message="No text found. This PDF might be scanned. Enable OCR or use a readable PDF.")

    # Clean text
    cleaned_text = clean_text(pdf_text)

    # Chunk and build FAISS index
    try:
        chunks = qa_utils.chunk_text_to_sentences(cleaned_text, max_chars=1000)
        doc_id = unique_name
        qa_utils.build_faiss_index_for_doc(doc_id, chunks)
    except Exception as idx_error:
        print("Index Error:", idx_error)
        return render_template("error.html", message="Failed to build search index.")

    # Generate summary
    try:
        summary = generate_summary(cleaned_text)
    except Exception as sum_error:
        print("Summarizer Error:", sum_error)
        summary = "Summary unavailable due to model error."

    # Render results page with summary + Q/A
    return render_template("results.html", doc_id=doc_id, summary=summary)


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json(force=True)
    doc_id = data.get("doc_id")
    question = data.get("question")

    if not doc_id or not question:
        return jsonify({"error": "Missing document ID or question."}), 400

    try:
        result = qa_utils.answer_question_by_retriever_reader(doc_id, question, top_k=4)
    except Exception as e:
        print("Q/A Error:", e)
        return jsonify({"error": "Q/A system failed. Try rephrasing your question."}), 500

    return jsonify({
        "answer": result.get("answer", ""),
        "score": result.get("score", 0.0),
        "source_chunk_idx": result.get("source_chunk_idx"),
        "context": result.get("context", "")
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # use_reloader=False prevents upload reload issues
