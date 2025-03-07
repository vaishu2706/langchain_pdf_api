from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
import os
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__)

# Include the PDF file path
PDF_FILE_PATH = "C:\\12n8\\pdfs\\Full-Moon.pdf"

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

@app.route("/extract-text", methods=["GET"])
def extract_text():
    try:
        loader = PyPDFLoader(PDF_FILE_PATH)
        documents = loader.load()
        extracted_text = " ".join([doc.page_content.replace("\uf0d8", "-").replace("\n", " ").strip() for doc in documents])
        extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
        
        return jsonify({"extracted_text": extracted_text})
    except Exception as e:
        return jsonify({"error": f"Failed to extract text: {str(e)}"})

@app.route("/search-text", methods=["GET"])
def search_text():
    keyword = request.args.get("keyword")
    if not keyword:
        return jsonify({"error": "Keyword parameter is required"}), 400
    
    try:
        loader = PyPDFLoader(PDF_FILE_PATH)
        documents = loader.load()
        extracted_text = " ".join([doc.page_content.replace("\uf0d8", "-").replace("\n", " ").strip() for doc in documents])
        extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
        
        keyword = keyword.strip().lower()
        matching_sentences = [sentence for sentence in extracted_text.split('.') if keyword in sentence.lower()]
        
        if matching_sentences:
            return jsonify({"message": "Keyword found", "matching_text": matching_sentences})
        else:
            return jsonify({"message": "No matching content found"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to search text: {str(e)}"}), 500

@app.route("/update-text", methods=["PUT"])
def update_text():
    data = request.get_json()
    old_text = data.get("old_text")
    new_text = data.get("new_text")
    
    if not old_text or not new_text:
        return jsonify({"error": "Both old_text and new_text are required"}), 400
    
    try:
        loader = PyPDFLoader(PDF_FILE_PATH)
        documents = loader.load()
        extracted_text = " ".join([doc.page_content.replace("\uf0d8", "-").replace("\n", " ").strip() for doc in documents])
        extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
        
        if old_text not in extracted_text:
            return jsonify({"error": "Text not found"}), 404
        
        updated_text = extracted_text.replace(old_text, new_text)
        return jsonify({"updated_text": updated_text})
    except Exception as e:
        return jsonify({"error": f"Failed to update text: {str(e)}"}), 500

@app.route("/delete-text", methods=["DELETE"])
def delete_text():
    data = request.get_json()
    text_to_delete = data.get("text_to_delete")
    
    if not text_to_delete:
        return jsonify({"error": "text_to_delete is required"}), 400
    
    try:
        loader = PyPDFLoader(PDF_FILE_PATH)
        documents = loader.load()
        extracted_text = " ".join([doc.page_content.replace("\uf0d8", "-").replace("\n", " ").strip() for doc in documents])
        extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
        
        
        if text_to_delete not in extracted_text:
            return jsonify({"error": "Text not found"}), 404
        
        updated_text = extracted_text.replace(text_to_delete, "")
        return jsonify({"updated_text": updated_text})
    except Exception as e:
        return jsonify({"error": f"Failed to delete text: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
