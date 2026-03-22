import os
from threading import Lock

import gradio as gr
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

ALERT_MESSAGE = "That's a geography question. However, I can tell you that Sahil Vishwakarma's portfolio doesn't directly mention his work experience in geography or politics."
ALERT_KEYWORDS = {"geography", "capital", "country", "continent", "president", "government", "politics", "policy", "law"}

FALLBACK_PHRASES = {
    "i don't know", "i do not know", "i'm not sure",
    "i am not sure", "i'm sorry", "apologies", "unable to answer",
}


class PortfolioChatBot:
    def __init__(self):
        self._chain = None
        self._llm = None
        self._initialized = False
        self._lock = Lock()

    def _initialize(self):
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY not set")

        self._llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=groq_key
        )

        documents = [
            "Sahil is a backend engineer focused on Django, microservices, and reliable APIs that scale.",
            "He works with AI tools like LangChain, vector databases, and RAG.",
            "He uses Kafka, Celery, caching, and observability tools.",
            "He builds AI developer tools and improves search systems.",
            "Available for backend roles and freelance work."
        ]

        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
        docs = splitter.create_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        self._chain = RetrievalQA.from_chain_type(
            llm=self._llm,
            retriever=vectorstore.as_retriever(),
        )

        self._initialized = True

    def ask(self, question: str) -> str:
        if not self._initialized:
            self._initialize()

        response = self._chain.run(question).strip()

        if any(p in response.lower() for p in FALLBACK_PHRASES):
            return self._fallback(question)

        return response

    def _fallback(self, question):
        prompt = f"Answer briefly: {question}"
        message = self._llm.predict_messages([HumanMessage(content=prompt)])
        return message.content.strip()


bot = PortfolioChatBot()

def chat(message, history):
    return bot.ask(message)


gr.ChatInterface(chat).launch()