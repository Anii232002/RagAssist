import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX", "ragassist")
index = pc.Index(INDEX_NAME)

# Use same model , Embeddings must match what we used in loaders.py
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def get_retriever():

     """Return a retriever from Pinecone index."""
     vectorstore = PineconeVectorStore(index=index,embedding=embeddings,text_key="text")
     return vectorstore.as_retriever(search_kwargs={"k": 3})

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

