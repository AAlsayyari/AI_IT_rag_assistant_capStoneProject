from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "qwen2.5:7b"
DB_NAME = str(Path(__file__).parent.parent / "vector_db_it")

# BIIABAAI/bge-m3
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
RETRIEVAL_K = 10

SYSTEM_PROMPT = """You are an intelligent Level 1 Technical Support Assistant for the Deanship of e-Transactions at King Saud University (KSU).
Your primary task is to help users and university staff solve technical issues and explain system usage steps concisely.

You must follow these strict rules:
1. Grounding: ONLY use the information provided in the Context below to answer the question. Do not use outside knowledge.
2. Fallback: If the Context does not contain the answer, say exactly: "عذراً، لا أملك معلومات حول هذا الموضوع في قاعدة المعرفة الحالية." (Sorry, I don't have information about this in the current knowledge base). Do not guess or hallucinate.
3. Tone: Keep your answers clear, step-by-step, and concise.
4. Language: Reply in the exact same language the user used (Arabic, English, or Mixed). If answering in Arabic, keep specific IT technical terms (e.g., 'Router', 'IP Address') in English if they appear that way in the context.
5. Forbidden Language: You MUST NEVER output Chinese characters. If you are unsure, default to Arabic.

Context:
{context}

Before answering, briefly analyze the context and the user's query in English within a <think>...</think> block to ensure accuracy. Then, provide your final response to the user outside the block.
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
llm = ChatOllama(temperature=0, model=MODEL)


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def rewrite_query(question: str, history: list[dict]) -> str:
    if not history:
        return question
        
    recent_history = history[-4:] 
    history_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in recent_history)
    
    rewrite_prompt = f"""You are a query rewriting engine for a King Saud University technical support Knowledge Base.
    Your task is to convert a conversational follow-up question into a fully formed, standalone semantic search query.

    Strict Rules:
    1. Contextualize: Replace pronouns (e.g., "how do I fix it") with the actual subject from the history.
    2. Format: Write a complete, natural sentence. Do not strip it down to short keywords, as this feeds a dense semantic vector database.
    3. Language: Keep the query in the exact same language as the user's current question (Arabic or English).
    4. Guardrails: Do NOT answer the question. Do NOT output  blocks.
    5. Output ONLY the rewritten query string.

    Conversation History:
    {history_text}
    
    Current Follow-up: {question}
    Standalone Query:"""
    
    response = llm.invoke([HumanMessage(content=rewrite_prompt)])
    return response.content.strip()

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    # Get the contextualized question for Chroma
    standalone_question = rewrite_query(question, history)
    
    # Search the vector database using ONLY the clean, standalone question
    docs = fetch_context(standalone_question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Pass the context and the original history to the main system prompt
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    
    # The final LLM call answers the user's original conversational question
    messages.append(HumanMessage(content=question))
    
    response = llm.invoke(messages)
    return response.content, docs