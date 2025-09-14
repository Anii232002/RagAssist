from .supabase_client import supabase
from flask import send_file, jsonify
import io
import mimetypes
BUCKET_NAME = "RagAssist"

def upload_document(file_path: str,filename:str):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # fallback

    with open(file_path, "rb") as f:
        res = supabase.storage.from_(BUCKET_NAME).upload(
            filename,
            f,
            {"content-type": mime_type}   # 👈 set correct MIME
        )
    return res

def list_documents():
    res = supabase.storage.from_(BUCKET_NAME).list()
    return [item['name'] for item in res]

def delete_document(file_name: str):
    res = supabase.storage.from_(BUCKET_NAME).remove([file_name])
    print(res)
    return res

def download_document(file_name: str):
    res = supabase.storage.from_(BUCKET_NAME).download(file_name)
    return res
def get_downloadable_url(file_name:str):
    res = supabase.storage.from_(BUCKET_NAME).get_public_url(file_name)
    return res
    
