from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Connect to your document database
persistent_directory = "db/chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Set up AI model
model = ChatOpenAI(model="gpt-4o")

# Store our conversation as messages
chat_history = []


def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Make the question standalone using chat history
    if chat_history:
        messages = [
            SystemMessage(content="Rewrite the new question to be standalone and searchable. Do not include conversation history in the output."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")

    else:
        search_question = user_question

    # Step 2: Retrieve relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")

    # Step 3: Build document text separately
    docs_text = "\n".join([f"- {doc.page_content}" for doc in docs])

    # Step 4: Build final RAG prompt
    combined_input = f"""
You are an AI assistant. Answer the user's question using ONLY the information from the provided documents.

User question:
{user_question}

Documents:
{docs_text}

If the documents do not contain enough information to answer, reply:
"I don't have enough information to answer that based on the provided documents."
"""

    # Step 5: Get final answer
    messages = [
        SystemMessage(content="You answer ONLY using the documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content

    # Step 6: Save conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"\nAnswer: {answer}")
    return answer


def start_chat():
    print("Ask me anything! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()