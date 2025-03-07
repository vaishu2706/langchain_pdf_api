from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
import os
import logging
import re
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

documents_db = {} 

@app.route("/upload", methods=["POST"])
def upload_pdf():
    data = request.get_json()
    file_path = data.get("file_path")
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "Invalid or missing file path"}), 400
    
    document_id = str(uuid.uuid4())
    documents_db[document_id] = file_path
    return jsonify({"document_id": document_id, "file_path": file_path})

def load_pdf(file_path):
    try:
        logger.info(f"Loading PDF from {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from PDF")
        return documents
    except FileNotFoundError:
        logger.error(f"PDF file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        return None

@app.route("/extract-text/<document_id>", methods=["GET"])
def extract_text(document_id):
    if document_id not in documents_db:
        return jsonify({"error": "Document not found"}), 404
    
    file_path = documents_db[document_id]
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        extracted_text = " ".join([doc.page_content.replace("\uf0d8", "-").replace("\n", " ").strip() for doc in documents])
        extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
        return jsonify({"document_id": document_id, "extracted_text": extracted_text})
    except Exception as e:
        return jsonify({"error": f"Failed to extract text: {str(e)}"})

@app.route("/search-text/<document_id>", methods=["GET"])
def search_text(document_id):
    keyword = request.args.get("keyword")
    if not keyword:
        return jsonify({"error": "Keyword parameter is required"}), 400
    
    if document_id not in documents_db:
        return jsonify({"error": "Document not found"}), 404
    try:
        file_path = documents_db[document_id]
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        extracted_text = " ".join([doc.page_content.replace("\uf0d8", "-").replace("\n", " ").strip() for doc in documents])
        extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
        
        keyword = keyword.strip().lower()
        matching_sentences = [sentence for sentence in extracted_text.split('.') if keyword in sentence.lower()]
        
        if matching_sentences:
            return jsonify({"document_id": document_id, "message": "Keyword found", "matching_text": matching_sentences})
        else:
            return jsonify({"document_id": document_id, "message": "No matching content found"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to search text: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
