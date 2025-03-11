from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from typing import List
from langchain_core.documents import Document
import os
import uuid

app = Flask(__name__)
os.environ["GOOGLE_API_KEY"] = "AIzaSyDzHec2q6RJoK0wKd2mrV-F2mLob8Tno34"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

@app.route("/upload", methods=['POST'])
def upload_file():
    try:
        data = request.get_json()
        file_path = data.get("file_path")
        question = data.get("question")
        if not file_path:
            return jsonify({"error": "No file path provided"}), 400
            
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)
        
        for doc in all_splits:
            doc.metadata["document_id"] = str(uuid.uuid4())
        
        vector_store = InMemoryVectorStore(embedding=embeddings)
        
        vector_store.add_documents(documents=all_splits)
        
        results = vector_store.similarity_search(question, k=3)
        
        if results:
            response = llm.invoke(f"Based on this text:'{results[0].page_content}' {question}")
        
            related_texts = []
            for idx, result in enumerate(results):
                related_texts.append({
                    "document_id": result.metadata.get("document_id"),
                    "chunk_text": result.page_content
                })
            
            return jsonify({
                "question": question,
                "answer": response.content,
                "relatedtext": related_texts
            })
        else:
            return jsonify({"error": "No relevant content found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
