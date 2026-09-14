from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "qwen2.5:7b"
DB_NAME = str(Path(__file__).parent.parent / "vector_db_it")

# BIIABAAI/bge-m3
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
RETRIEVAL_K = 10

SYSTEM_PROMPT = """You are an intelligent Level 1 Technical Support Assistant for the Deanship of e-Transactions at King Saud University (KSU).

CRITICAL INSTRUCTIONS:
1. Permitted Alphabets: You MUST strictly write using ONLY the Arabic alphabet and the English alphabet. You are absolutely prohibited from generating any Asian characters (Chinese, Japanese, Korean, etc.) under any circumstances.
2. Grounding: If the user asks a technical question, base your answer SOLELY on the Context below.
3. Small Talk / Greetings: If the user asks general questions like "كيف تخدمني", "مرحبا", or makes small talk, immediately reply with: "أنا خبير الدعم الفني لجامعة الملك سعود. يمكنني مساعدتك في حل المشاكل التقنية واستخدام الأنظمة الإلكترونية. كيف يمكنني مساعدتك اليوم؟"
4. Fallback: If the Context does not contain the technical answer, reply EXACTLY with: "عذراً، لا أملك معلومات حول هذا الموضوع في قاعدة المعرفة الحالية."
5. Tone: Keep technical answers step-by-step and concise. Use English for specific IT terms (e.g., 'Router', '2FA').

Context:
{context}

Answer directly without any introductory filler, translation notes, or internal thinking text.
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
llm = ChatOllama(
    model=MODEL, 
    temperature=0, 
    # Force the model to stop if it tries to hallucinate the next turn
    stop=["<|im_end|>", "<|im_start|>", "user", "User:", "Human:"] 
)


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

# 1. Define a strict LangChain Chat Template
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    # Get the context using the rewrite function 
    standalone_question = rewrite_query(question, history) if history else question
    docs = fetch_context(standalone_question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Safely convert Gradio's history dictionaries into LangChain message objects
    langchain_history = convert_to_messages(history)
    
    # 2. Pipe the formatted prompt directly into the LLM
    chain = qa_prompt | llm
    
    # 3. Invoke the chain with our mapped variables
    response = chain.invoke({
        "context": context,
        "chat_history": langchain_history,
        "question": question
    })
    
    return response.content, docs