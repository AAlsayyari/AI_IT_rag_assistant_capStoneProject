from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "qwen2.5:14b"
DB_NAME = str(Path(__file__).parent.parent / "vector_db_it")

# BIIABAAI/bge-m3
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"local_files_only": True})
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
RETRIEVAL_K = 10

SYSTEM_PROMPT = """
CRITICAL RULE: Always detect the language of the user's latest message and reply strictly in that same language. If the user writes in Arabic, you MUST reply in Arabic. If the user writes in English, you MUST reply in English. Never answer an Arabic message in English.

قاعدة هامة: يجب الرد دائماً بنفس لغة المستخدم. إذا تحدث المستخدم بالعربية، أجب باللغة العربية حصراً.

Role and Instructions:
- You are a bilingual technical support assistant for the Deanship of e-Transactions at King Saud University (عمادة التعاملات الإلكترونية بجامعة الملك سعود).
- Assist users and university staff in resolving technical issues concisely based on official user manuals and FAQs.
- You only support Arabic and English. If a user writes in any other language, inform them that you only support Arabic and English.
- If you do not know the answer, state that you do not know.

Context:
{context}
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


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs