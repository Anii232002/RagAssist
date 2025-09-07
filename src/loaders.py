import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore



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

def rebuild_vectorstore(folder_path: str):
    """Rebuild Pinecone index from all files in folder (deletes old data)."""
    import shutil

    index = pc.Index(INDEX_NAME)
    index.delete(delete_all=True)

    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath):
            docs = load_and_split(fpath)
            PineconeVectorStore.from_documents(docs, embeddings, index_name=INDEX_NAME)
    print("Pinecone index rebuilt successfully.")

if __name__ == "__main__":
    # Example: add notes.txt into DB
    test_file = os.path.join(os.path.dirname(__file__), "../data/notes.txt")
    add_to_vectorstore(test_file)