from . import config
import os
from flask import Flask, request, jsonify
from .rag_chain import ask
from .loaders import add_to_vectorstore, rebuild_vectorstore
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__),"storage")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/ask",methods=["POST"])
def ask_question():
    """API endpoint to query the RAG bot."""
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error","Query is required"}),400
    
    try:
        result = ask(query)
        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/ingest",methods=["POST"])
def ingest():
    
    if "file" not in request.files:
        return jsonify({"error":"No files uploaded"}),400
    file = request.files["file"]

    if file.filename =="":
         return jsonify({"error": "Empty filename"}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        chunks = add_to_vectorstore(file_path)
        return jsonify({"message": f"File ingested successfully", "chunks": chunks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/list",methods=["GET"])
def list_docs():
    """List all uploaded documents."""
    try:
        files = os.listdir(app.config["UPLOAD_FOLDER"])
        return jsonify({"documents": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/delete",methods=["POST"])
def delete_doc():
    """Delete a document and re-index."""
    try:
        data = request.get_json()
        filename = data.get("filename")

        if not filename:
            return jsonify({"error": "File name required"})
        
        file_path = os.path.join(app.config["UPLOAD_FOLDER"],filename)

        if not os.path.exists(file_path):
            return jsonify({"error":"File not found"}), 404
        
        os.remove(file_path)

        rebuild_vectorstore(app.config["UPLOAD_FOLDER"])

        return jsonify({"message": f"{filename} deleted and index updated."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)