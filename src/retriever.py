import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIRECTORY = os.getenv(
    "PERSIST_DIRECTORY",
     os.path.join(os.path.dirname(__file__), "../storage/chroma_store")
)

# Use same model , Embeddings must match what we used in loaders.py
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def get_retriever():
     """Connects to Chroma DB and returns a retriever object."""
     db = Chroma(
          persist_directory=PERSIST_DIRECTORY,
          embedding_function=embeddings
     )

    #Now search in the db
     retriever = db.as_retriever(
          search_type="similarity",
          search_kwargs={"k": 3} #no of chunks to fetch
     )

     return retriever

#TEST below only..
#retriever.get_relevant_documents(query)
#the fn get_relevant_documents is a fn which Chroma provides, since we are returning its instance back

if __name__ == "__main__":
    # Quick test
    retriever = get_retriever()
    query = "What did I learn about RAG?"
    results = retriever.get_relevant_documents(query)

    print("🔎 Retrieved Chunks:")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Chunk {i} ---\n{doc.page_content[:300]}")

