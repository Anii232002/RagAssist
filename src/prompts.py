from langchain.prompts import ChatPromptTemplate

# System instructions
SYSTEM_INSTRUCTIONS = (
    "You are MyKnowledgeBot, a helpful assistant that answers ONLY using the "
    "provided context from the user's personal notes. "

)

#Prompt template

NOTES_PROMPT = ChatPromptTemplate.from_messages(
    [
         ("system", SYSTEM_INSTRUCTIONS),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
    ]
)