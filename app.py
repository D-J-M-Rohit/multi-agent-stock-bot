import os, logging, sqlite3, redis
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

# LangChain & agents ----------------------------------------------------------
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma   # used inside rag_agent
from agents import (
    stock_price_agent,
    news_agent,
    earnings_agent,
    market_summary_agent,
    rag_agent,
)

# Auth module -----------------------------------------------------------------
import auth

# ─── persistence: SQLite for chat log & Redis for Q/A cache ──────────────────
CHAT_DB = "chat_history.db"

def _get_conn():
    """Return a connection to the SQLite DB (creates file if missing)."""
    return sqlite3.connect(CHAT_DB, check_same_thread=False)

def init_chat_db():
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                username  TEXT NOT NULL,
                role      TEXT CHECK(role IN ('user', 'assistant')) NOT NULL,
                text      TEXT NOT NULL,
                ts        DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

def fetch_chat(username: str, limit: int = 200):
    """Return the last `limit` chat messages for `username`."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, text FROM chats WHERE username = ? ORDER BY id ASC LIMIT ?",
            (username, limit),
        ).fetchall()
    return [{"role": r, "text": t} for r, t in rows]

def save_chat(username: str, role: str, text: str):
    """Persist a single chat message."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO chats (username, role, text, ts) VALUES (?, ?, ?, ?)",
            (username, role, text, datetime.utcnow()),
        )

# Redis helpers ---------------------------------------------------------------
load_dotenv()  # ensure env vars for Redis are loaded early

def _init_redis():
    try:
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True,
        )
        client.ping()
        logging.info("Connected to Redis – caching enabled ✅")
        return client
    except Exception as e:
        logging.warning(f"Redis unavailable – caching disabled. ({e})")
        return None

redis_client = _init_redis()
_DEFAULT_TTL = 60 * 60 * 24  # 24h

def _cache_key(user: str, query: str) -> str:
    return f"qa:{user}:{query.strip().lower()}"

def get_cached_answer(user: str, query: str):
    if not redis_client:
        return None
    try:
        return redis_client.get(_cache_key(user, query))
    except Exception:
        return None

def set_cached_answer(user: str, query: str, answer: str, ttl: int = _DEFAULT_TTL):
    if not redis_client:
        return
    try:
        redis_client.set(_cache_key(user, query), answer, ex=ttl)
    except Exception:
        logging.debug("Redis set failed; continuing without cache.")

# ─── basic setup & keys ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY – add it to .env or your shell env")
    st.stop()

# initialise auth and chat db
auth.init_db()
init_chat_db()

# ─── construct shared LLM + agents ───────────────────────────────────────────
_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

stock_agent = stock_price_agent.create_agent(_llm)
news_agent_inst = news_agent.create_agent(_llm)
financials_agent = earnings_agent.create_agent(_llm)
market_agent = market_summary_agent.create_agent(_llm)

# ─── routing helper ──────────────────────────────────────────────────────────

def route_query(query: str) -> str:
    """Decide which specialised agent(s) should answer `query` and merge partial answers."""
    q = query.lower()
    wants_price = any(w in q for w in ["price", "quote", "trading at"])
    wants_news = "news" in q or "headline" in q
    wants_financial = any(w in q for w in ["earnings", "financial", "revenue"])
    wants_market = any(w in q for w in ["market", "index", "indices", "dow", "nasdaq", "s&p"])

    # crude ticker detector (all-caps token)
    ticker = next((t for t in query.split() if t.isalpha() and t.isupper()), None)
    if ticker and not (wants_price or wants_news or wants_financial or wants_market):
        wants_price = True

    # multi‑part questions like “AAPL price and latest news”
    if " and " in q:
        parts = [p.strip() for p in q.split(" and ") if p.strip()]
        return "\n\n".join(route_query(p) for p in parts)

    responses = []
    try:
        if wants_price:
            responses.append(stock_agent.run(ticker or query))
        if wants_news:
            responses.append(news_agent_inst.run(ticker or query))
        if wants_financial:
            responses.append(financials_agent.run(ticker or query))
        if wants_market and not wants_news:
            responses.append(market_agent.run(query))
        if not responses:  # fallback to RAG KB
            responses.append(rag_agent.answer_query(query))
    except Exception as e:
        logging.exception(e)
        return "Sorry, I hit a problem while processing that."

    if len(responses) == 1:
        return responses[0]

    out, idx = [], 0
    if wants_price:
        out.append("**Price:** " + responses[idx]); idx += 1
    if wants_news:
        out.append("**News:** " + responses[idx]); idx += 1
    if wants_financial:
        out.append("**Financials:** " + responses[idx]); idx += 1
    if wants_market and not wants_news:
        out.append("**Market:** " + responses[idx]); idx += 1
    return "\n\n".join(out)

# ─── Streamlit page config ───────────────────────────────────────────────────
st.set_page_config(page_title="💹 Multi-Agent Stock Chatbot", page_icon="💹", layout="wide")

# ─── Auth sidebar ────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

with st.sidebar:
    st.title("🔐  Account")
    if not st.session_state.logged_in:
        action = st.radio("Action", ["Login", "Register"])
        if action == "Login":
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Login"):
                if auth.verify_login(u, p):
                    st.session_state.logged_in = True
                    st.session_state.username = u
                    st.session_state.messages = fetch_chat(u)
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            st.stop()

        # Registration
        u = st.text_input("Username")
        e = st.text_input("Email")
        n = st.text_input("Full name")
        p1 = st.text_input("Password", type="password")
        p2 = st.text_input("Confirm", type="password")
        if st.button("Register"):
            if not all([u, e, n, p1]):
                st.error("Please fill in all fields.")
            elif p1 != p2:
                st.error("Passwords don’t match.")
            elif auth.register_user(u, e, n, p1):
                st.success("Registered – please log in.")
            else:
                st.error("Username already exists.")
        st.stop()

    display_name = auth.get_full_name(st.session_state.username) or st.session_state.username
    st.success(f"Logged in as **{display_name}**")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.messages = []
        st.rerun()

# ─── Session state helpers ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = fetch_chat(st.session_state.username) if st.session_state.logged_in else []
if "clear_next" not in st.session_state:
    st.session_state.clear_next = False
if st.session_state.clear_next:
    st.session_state.query_input = ""
    st.session_state.clear_next = False

# ─── UI: chat history ────────────────────────────────────────────────────────
st.title("📊 Multi-Agent Stock Chatbot")
for m in st.session_state.messages:
    who = "User" if m["role"] == "user" else "Assistant"
    st.markdown(f"**{who}:** {m['text']}")

st.markdown("---")

# ─── UI: input widget + button ───────────────────────────────────────────────
query_text = st.text_input(
    "Your question:", key="query_input", placeholder="e.g. What’s the outlook for AAPL earnings?"
)

analyze_clicked = st.button("Analyze")

# ─── Main logic: check cache ➜ run agents ➜ cache ➜ show ─────────────────────
if analyze_clicked and query_text:
    user_query = query_text

    # store & echo user
    st.session_state.messages.append({"role": "user", "text": user_query})
    save_chat(st.session_state.username, "user", user_query)
    st.markdown(f"**User:** {user_query}")

    # check Redis cache first
    answer = get_cached_answer(st.session_state.username, user_query)
    if answer:
        logging.info("Redis hit – skipping analysis ✨")
    else:
        logging.info("Redis miss – running agents …")
        answer = route_query(user_query)
        set_cached_answer(st.session_state.username, user_query, answer)

    # store & echo assistant
    st.session_state.messages.append({"role": "assistant", "text": answer})
    save_chat(st.session_state.username, "assistant", answer)
    st.markdown(f"**Assistant:** {answer}")

    # clear input on next refresh
    st.session_state.clear_next = True
    st.rerun()
