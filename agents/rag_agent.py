# rag_agent.py  ──────────────────────────────────────────────────────────────
import os
import logging
from typing import List

from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# ─── environment / logging ────────────────────────────────────────────────
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not found – export it or put it in your .env file."
    )

# ─── 1. Load raw documents (.txt + .pdf) ───────────────────────────────────
docs_dir     = "data/sample_docs"
persist_dir  = "data/vectorstore"
all_docs: List[Document] = []

splitter = CharacterTextSplitter(chunk_size=1_000, chunk_overlap=100)
supported_exts = {".txt", ".pdf"}

if os.path.isdir(docs_dir):
    for fname in os.listdir(docs_dir):
        fpath = os.path.join(docs_dir, fname)
        ext   = os.path.splitext(fname)[1].lower()

        if ext not in supported_exts:
            continue

        try:
            if ext == ".txt":
                loader_docs = TextLoader(fpath, encoding="utf-8").load()
            else:                      # PDF
                loader_docs = PyPDFLoader(fpath).load()

            # split every doc into smaller chunks
            for d in loader_docs:
                for chunk in splitter.split_text(d.page_content):
                    all_docs.append(Document(page_content=chunk, metadata=d.metadata))

        except Exception as e:
            logger.warning(f"⚠️  Skipped {fname} ({e})")

else:
    logger.warning(f"Directory {docs_dir} does not exist – no KB docs loaded.")

# ─── 2. Build / load the vector store ──────────────────────────────────────
embedding_fn = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

if os.path.isdir(persist_dir) and os.listdir(persist_dir):
    # Existing index – just load it (fast)
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_fn,
    )
else:
    # First-time build
    if not all_docs:
        raise RuntimeError(
            "No documents were loaded – please add .txt or .pdf files to "
            f"{docs_dir} before starting."
        )

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding_fn,
        persist_directory=persist_dir,
    )
    vectorstore.persist()      # save to disk

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ─── 3. Retrieval-augmented QA chain ───────────────────────────────────────
qa_llm   = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
qa_chain = RetrievalQA.from_chain_type(
    llm        = qa_llm,
    chain_type = "stuff",
    retriever  = retriever,
)

# ─── 4. Public helper --------------------------------------------------------
def answer_query(query: str) -> str:
    """
    Query the knowledge base with semantic search + LLM synthesis.
    """
    try:
        answer = qa_chain.run(query).strip()
        return answer or "I'm sorry, I don't have information on that topic."
    except Exception as e:
        logger.error(f"RAG failure: {e}")
        return "I'm sorry, I cannot find information on that."

# Optional CLI test
if __name__ == "__main__":
    while True:
        q = input("\nAsk: ")
        print(answer_query(q))
