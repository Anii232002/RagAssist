from . import config
import os
from flask import Flask, request, jsonify
from .rag_chain import ask,search_docs
from .loaders import add_to_vectorstore, rebuild_vectorstore
from werkzeug.utils import secure_filename
from .cloud_storage import delete_document,list_documents,upload_document,download_document,get_downloadable_url
from dotenv import load_dotenv
from pathlib import Path
import tempfile
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
            "answer": result["answer"]
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

    try:
        with tempfile.NamedTemporaryFile(delete=False)as tmp_file:
            file.save(tmp_file.name)
            upload_document(tmp_file.name,filename)
            tmp_file_path = tmp_file.name
        
        os.remove(tmp_file.name)

        file_bytes = download_document(filename)
        with tempfile.NamedTemporaryFile(delete=False,suffix=Path(filename).suffix) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        add_to_vectorstore(tmp_file_path)
        os.remove(tmp_file_path)
        return jsonify({"message": "File ingested successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/list",methods=["GET"])
def list_docs():
    """List all uploaded documents."""
    try:
        files = list_documents()
        return jsonify({"documents": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/delete",methods=["POST"])
def delete_doc():
    """Delete a document from Supabase and re-index."""
    try:
        data = request.get_json()
        filename = data.get("filename")

        if not filename:
            return jsonify({"error": "File name required"})
        
        if filename.startswith("http"):
            filename = filename.split("/")[-1]
        
        delete_document(filename)
        
        rebuild_vectorstore()

        return jsonify({"message": f"{filename} deleted and index updated."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/download",methods=["POST"])
def download_doc():
    """ Returns a signed URL for downloading a file from Supabase Storage.
    Expects JSON body: { "filename": "your_file.txt" }"""
    try:
        data = request.get_json()
        filename = data.get("filename")

        if not filename:
            return jsonify({"error":"File name is required"})
        
        return get_downloadable_url(filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/search",methods=["POST"])
def search_doc():
    try:
        data = request.get_json()
        query = data.get("query")
        print(query)

        if not query:
            return jsonify({"error": "Query not found"})
        raw_results = search_docs(query)

        normalized = []
        if raw_results is None:
            raw_results = []
        
        for item in raw_results:
            if isinstance(item, dict):
                normalized.append({
                    "file": item.get("file") or item.get("source") or item.get("doc"),
                    "page": item.get("page") or item.get("page_number") or item.get("p"),
                    "excerpt": item.get("excerpt") or item.get("snippet") or item.get("text") and item.get("text")[:400],
                    "score": item.get("score") or item.get("similarity") or 0.0,
                    "chunk_id": item.get("chunk_id") or item.get("id"),
                    "content": item.get("text") or item.get("content") or item.get("excerpt"),
                })
            # if a tuple/list (e.g. (doc, page, score, snippet))
            elif isinstance(item, (list, tuple)):
                # try to map reasonable positions: (file, page, score, excerpt, id)
                file = item[0] if len(item) > 0 else None
                page = item[1] if len(item) > 1 else None
                score = item[2] if len(item) > 2 else 0.0
                excerpt = item[3] if len(item) > 3 else None
                chunk_id = item[4] if len(item) > 4 else None
                normalized.append({
                    "file": file,
                    "page": page,
                    "excerpt": excerpt,
                    "score": float(score) if score is not None else 0.0,
                    "chunk_id": chunk_id,
                    "content": excerpt
                })
            else:
                # fallback: stringify object
                s = str(item)
                normalized.append({
                    "file": None,
                    "page": None,
                    "excerpt": s[:400],
                    "score": 0.0,
                    "chunk_id": None,
                    "content": s
                })

        return jsonify({"results": normalized}), 200
        print(res)
        return jsonify()
    except Exception as e:
        return jsonify({"error": str(e)})
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)