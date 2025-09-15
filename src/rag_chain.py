from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from .retriever import get_retriever

#Load LLM
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.5)
#temp = 0 for precise factual answer, more strict

#Load retriever
retriever = get_retriever()

# Creating a RAG Pipeline 

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = retriever,
    chain_type="stuff",
    return_source_documents=True
)

def ask(query:str):
    """Ask a question to the RAG system and return answer"""
    result = qa_chain.invoke({"query":query})
    return {
        "answer": result["result"]
    }

def search_docs(query):
    results = retriever.get_relevant_documents(query)
    return results

##TEST

if __name__ == "__main__":
    query = "What did I learn about RAG?"
    response = ask(query)

    print("\n💡 Answer:")
    print(response["answer"])

    print("\n📂 Sources:")
    for src in response["sources"]:
        print(src)