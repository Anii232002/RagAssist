import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


PERSIST_DIRECTORY = os.getenv(
    "PERSIST_DIRECTORY",
    os.path.join(os.path.dirname(__file__),"../storage/chroma_store")
)

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

     db = Chroma(
          persist_directory=PERSIST_DIRECTORY,
          embedding_function=embeddings
     )

     db.add_documents(docs)
     db.persist()
     print(f"Added {len(docs)} chunks from {file_path} to Chroma DB.")

def rebuild_vectorstore(folder_path: str):
    """Clear Chroma DB and rebuild from all files in folder."""
    import shutil

    #delete olde Chroma Store

    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

    db = Chroma(persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings)
    
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath):
            docs = load_and_split(fpath)
            db.add_documents(docs)

    db.persist()
    print("Vector store rebuilt successfully.")

if __name__ == "__main__":
    # Example: add notes.txt into DB
    test_file = os.path.join(os.path.dirname(__file__), "../data/notes.txt")
    add_to_vectorstore(test_file)