import os
import logging
from pathlib import Path
from threading import Lock

import gradio as gr
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── PDF path ───────────────────────────────────────────────────────────────────
PDF_PATH = Path(__file__).parent / "portfolio.pdf"

# ── Prompts ────────────────────────────────────────────────────────────────────
REPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ("human",
     "Given the conversation above, generate a concise standalone search query "
     "that captures what the user is asking. Return only the query, nothing else."),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional AI assistant representing Sahil Vishwakarma's portfolio. "
     "Answer questions about Sahil's skills, experience, projects, and background "
     "using ONLY the context below. Be concise, friendly, and professional.\n\n"
     "If a question is unrelated to Sahil or his work, respond with:\n"
     "'I'm here to answer questions about Sahil's portfolio and skills. "
     "For other topics, feel free to visit his GitHub or reach out directly.'\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# ── PDF loader ─────────────────────────────────────────────────────────────────
def load_pdf(pdf_path: Path) -> list:
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Portfolio PDF not found at '{pdf_path}'. "
            "Upload 'portfolio.pdf' to your Hugging Face Space repo alongside app.py."
        )

    logger.info("Loading PDF: %s", pdf_path)
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    if not pages:
        raise ValueError(
            "PDF loaded but no readable text found. "
            "Scanned/image-based PDFs require OCR — export yours as a text-based PDF."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(pages)
    logger.info("PDF → %d chunks from %d pages.", len(chunks), len(pages))
    return chunks


# ── Chatbot ────────────────────────────────────────────────────────────────────
class PortfolioChatBot:
    MAX_INPUT_LENGTH = 500

    def __init__(self, pdf_path: Path):
        self._pdf_path = pdf_path
        self._chain = None
        self._lock = Lock()

    def _build_chain(self):
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it as a Space secret in your HF Spaces settings."
            )

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.2,
            api_key=groq_key,
            max_tokens=512,
        )

        docs = load_pdf(self._pdf_path)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, REPHRASE_PROMPT
        )
        answer_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
        self._chain = create_retrieval_chain(history_aware_retriever, answer_chain)
        logger.info("PortfolioChatBot ready.")

    def _ensure_ready(self):
        if self._chain is None:
            with self._lock:
                if self._chain is None:
                    self._build_chain()

    def ask(self, question: str, chat_history: list) -> str:
        question = question.strip()
        if not question:
            return "Please type a question."
        if len(question) > self.MAX_INPUT_LENGTH:
            return (
                f"Your message is too long (>{self.MAX_INPUT_LENGTH} chars). "
                "Please keep it concise."
            )
        try:
            self._ensure_ready()
            result = self._chain.invoke({
                "input": question,
                "chat_history": chat_history,
            })
            answer = result.get("answer", "").strip()
            return answer or "I'm not sure about that — could you rephrase?"
        except (FileNotFoundError, ValueError, EnvironmentError) as exc:
            logger.error("Setup error: %s", exc)
            return f"⚠️ Setup error: {exc}"
        except Exception:
            logger.exception("Unexpected error.")
            return "Sorry, something went wrong. Please try again."


# ── Global instance ────────────────────────────────────────────────────────────
bot = PortfolioChatBot(pdf_path=PDF_PATH)


# ── Gradio handler ─────────────────────────────────────────────────────────────
def chat(message: str, history: list) -> str:
    # Gradio 6.x with type="messages" sends history as list of {"role": ..., "content": ...} dicts
    lc_history = []
    for msg in history:
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_history.append(AIMessage(content=msg["content"]))
    return bot.ask(message, lc_history)


# ── Gradio UI ──────────────────────────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=chat,
    title="Sahil Vishwakarma — Portfolio Assistant",
    description=(
        "Ask me anything about Sahil's skills, experience, or how to get in touch. "
        "Powered by Groq LLM + LangChain RAG."
    ),
    examples=[
        "What technologies does Sahil work with?",
        "Is Sahil available for freelance work?",
        "Tell me about his projects.",
        "How can I contact Sahil?",
    ],
)

if __name__ == "__main__":
    demo.launch()
