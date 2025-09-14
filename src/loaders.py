import os
import tempfile
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from .cloud_storage import list_documents,download_document




pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX", "ragassist")

index = pc.Index(INDEX_NAME)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=100,
    add_start_index=True
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def load_and_split(file_path:str):
    """Loads a text file and splits it into chunks."""
    ext = Path(file_path).suffix.lower()

    # Select loader based on extension
    if ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    documents = loader.load()
    return text_splitter.split_documents(documents)

def add_to_vectorstore(file_path:str):
     """Load a file, create embeddings, and add them to Chroma DB."""
     docs = load_and_split(file_path)

     PineconeVectorStore.from_documents(docs, embeddings, index_name=INDEX_NAME)
     print(f"Added {len(docs)} chunks from {file_path} to Pinecone.")

def rebuild_vectorstore():
    """Rebuild Pinecone index from all files in folder (deletes old data)."""

    index.delete(delete_all=True)
    print(INDEX_NAME)

    docs_list = list_documents()
    for fname in docs_list:
        file_bytes = download_document(fname)
        ext = Path(fname).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        docs = load_and_split(tmp_file_path)
        PineconeVectorStore.from_documents(docs,embeddings,index_name=INDEX_NAME)

    print("Pinecone index rebuilt successfully.")

if __name__ == "__main__":
    # Example: add notes.txt into DB
    test_file = os.path.join(os.path.dirname(__file__), "../data/notes.txt")
    add_to_vectorstore(test_file)