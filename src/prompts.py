from langchain.prompts import ChatPromptTemplate

# System instructions
SYSTEM_INSTRUCTIONS = (
    "You are MyKnowledgeBot, a helpful assistant that answers ONLY using the "
    "provided context from the user's personal notes. "
    "If the answer is not in the context, say you don't know. "
    "Always produce concise, factual answers and include short inline citations "
    "like (source: filename:page)."
)

#Prompt template

NOTES_PROMPT = ChatPromptTemplate.from_messages(
    [
         ("system", SYSTEM_INSTRUCTIONS),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
    ]
)