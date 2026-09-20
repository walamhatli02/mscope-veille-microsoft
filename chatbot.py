import streamlit as st
import json
import os
import time
import threading
import unicodedata
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from scraper import lancer_collecte
from agent import lancer_agent

load_dotenv()

def get_groq_model():
    preferred = os.getenv("GROQ_MODEL")
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend([
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ])
    for model in candidates:
        if model:
            return model
    return "openai/gpt-oss-20b"

VECTORSTORE_PATH = "vectorstore"
LLM_MODEL = get_groq_model()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RAPPORT_PATH = "data/rapport_agent.json"
ARTICLES_PATH = "data/articles.json"
INSOMEA_PATH = "insomea_info.txt"

st.set_page_config(page_title="MsScope — Veille Microsoft", page_icon="🤖", layout="wide")

if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Accueil"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_exemple" not in st.session_state:
    st.session_state.question_exemple = None

st.markdown("""
<style>
  * { font-family: 'Segoe UI', sans-serif; }
  .stApp { background: #f3f2f1; }
  .block-container { padding: 0 !important; max-width: 100% !important; }
  section[data-testid="stSidebar"] { display: none !important; }
  header[data-testid="stHeader"] { display: none !important; }
  .stButton > button {
    background-color: #0078D4 !important;
    color: white !important;
    border: none !important;
    border-radius: 2px !important;
    font-size: 13px !important;
  }
  .stButton > button:hover { background-color: #106EBE !important; }
  .stChatMessage { padding: 0 32px !important; }
  div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) {
    background: white;
    border-bottom: 1px solid #e1dfdd;
    padding: 0 32px;
    gap: 0 !important;
  }
  div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    border-bottom: 3px solid transparent !important;
    padding: 14px 24px !important;
    color: #323130 !important;
    font-size: 14px !important;
    box-shadow: none !important;
    height: auto !important;
  }
  div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
    color: #0067b8 !important;
    border-bottom-color: #0067b8 !important;
    background: #f9f8f7 !important;
  }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
def charger_articles():
    if os.path.exists(ARTICLES_PATH):
        with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def charger_rapport():
    if not os.path.exists(RAPPORT_PATH):
        return []
    try:
        with open(RAPPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def parse_date_article(date_str):
    if not date_str:
        return datetime.min
    date_str = str(date_str).strip()
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S",
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt
        except ValueError:
            pass
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except ValueError:
        return datetime.min

def preprocess_question(text: str) -> str:
    if not text:
        return ""
    s = str(text).lower().strip()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    s = ' '.join(s.split())
    s = re.sub(r'[!?]{2,}', '?', s)
    s = re.sub(r'[\.]{2,}', '.', s)
    return s

# ============================================================
# RAG EN MEMOIRE
# ============================================================
@st.cache_resource
def charger_rag():
    import chromadb
    import time
    from langchain_chroma import Chroma

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    client = chromadb.EphemeralClient()

    vs_microsoft = Chroma(client=client, collection_name="microsoft_articles", embedding_function=embeddings)
    vs_insomea = Chroma(client=client, collection_name="insomea_info", embedding_function=embeddings)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    if os.path.exists(ARTICLES_PATH):
        with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
            articles = json.load(f)
        docs = []
        for article in articles:
            chunks = splitter.split_text(f"{article['titre']}\n\n{article.get('contenu', '')}")
            for i, chunk in enumerate(chunks):
                docs.append(Document(page_content=chunk, metadata={
                    "source": article.get("source", ""),
                    "titre": article.get("titre", ""),
                    "lien": article.get("lien", ""),
                    "date": article.get("date", ""),
                    "chunk": i
                }))
        if docs:
            vs_microsoft.add_documents(docs)

    if os.path.exists(INSOMEA_PATH):
        with open(INSOMEA_PATH, "r", encoding="utf-8") as f:
            contenu = f.read()
        chunks = splitter.split_text(contenu)
        docs_ins = [Document(page_content=chunk, metadata={
            "source": "INSOMEA",
            "titre": "Informations et offres INSOMEA",
            "lien": "https://insomea.com",
            "date": "2026-01-01",
            "chunk": i
        }) for i, chunk in enumerate(chunks)]
        if docs_ins:
            vs_insomea.add_documents(docs_ins)

    retriever_microsoft = vs_microsoft.as_retriever(search_kwargs={"k": 4})
    retriever_insomea = vs_insomea.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""Tu es InsoBot, assistant expert Microsoft pour l'équipe commerciale d'INSOMEA en Tunisie.
Tu dois répondre à TOUTES les questions liées à Microsoft, cloud, technologie et INSOMEA.
Réponds TOUJOURS en français, même si la question est mal formulée.
Si tu n'as pas l'info exacte, utilise tes connaissances générales Microsoft.
Réponds de façon claire et concise (2 à 5 phrases).

Informations INSOMEA: {context_insomea}
Actualités Microsoft: {context_microsoft}
Question: {question}
Réponse en français:""")

    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.2, max_tokens=200)

    def formater(docs):
      return "\n".join([f"{doc.metadata.get('titre','')}: {doc.page_content[:100]}" for doc in docs[:2]])

    def run_rag(question):
        try:
            docs_microsoft = retriever_microsoft.invoke(question)
            docs_insomea = retriever_insomea.invoke(question)
        except Exception:
            docs_microsoft = []
            docs_insomea = []

        payload = {
            "context_microsoft": formater(docs_microsoft) if docs_microsoft else "Pas d'actualités disponibles.",
            "context_insomea": formater(docs_insomea) if docs_insomea else "Pas d'infos INSOMEA.",
            "question": question
        }

        chain = prompt | llm | StrOutputParser()

        response = None
        for attempt in range(3):
            try:
                response = chain.invoke(payload)
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    time.sleep((attempt + 1) * 5)
                else:
                    raise e

        if not response:
            response = "Limite de requêtes atteinte. Veuillez réessayer dans quelques secondes."

        chunk_size = 20
        for i in range(0, len(response), chunk_size):
            yield response[i:i + chunk_size], docs_microsoft

    return run_rag

# ============================================================
# BACKGROUND TASKS
# ============================================================
def refresh_background():
    try:
        lancer_collecte()
        lancer_agent()
    except Exception:
        pass

def demarrer_refresh_periodique(interval_seconds=3600):
    if "periodic_refresh_started" in st.session_state and st.session_state.periodic_refresh_started:
        return
    st.session_state.periodic_refresh_started = True
    def boucle_periodique():
        while True:
            try:
                refresh_background()
            except Exception:
                pass
            time.sleep(interval_seconds)
    th = threading.Thread(target=boucle_periodique, daemon=True)
    th.start()

def lancer_rapport_si_besoin():
    if "rapport_agent_started" not in st.session_state:
        st.session_state.rapport_agent_started = False
    if st.session_state.rapport_agent_started:
        return
    st.session_state.rapport_agent_started = True
    thread = threading.Thread(target=lambda: lancer_agent(force=True), daemon=True)
    thread.start()

if "auto_refresh_done" not in st.session_state:
    st.session_state.auto_refresh_done = False
if not st.session_state.auto_refresh_done:
    demarrer_refresh_periodique(interval_seconds=600)
    st.session_state.auto_refresh_done = True

if not st.session_state.get("rapport_agent_started", False):
    lancer_rapport_si_besoin()

articles_bruts = charger_articles()
articles = sorted(articles_bruts, key=lambda x: parse_date_article(x.get("date", "")), reverse=True)
rapport = charger_rapport()

# ============================================================
# TOPBAR
# ============================================================
st.markdown("""
<div style="background:white;border-bottom:1px solid #e1dfdd;padding:0 32px;display:flex;align-items:center;justify-content:space-between;height:48px;position:sticky;top:0;z-index:100;box-shadow:0 1px 4px rgba(0,0,0,0.08);">
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="display:flex;gap:2px;flex-wrap:wrap;width:20px;height:20px;">
      <span style="background:#f25022;width:9px;height:9px;display:block;"></span>
      <span style="background:#7fba00;width:9px;height:9px;display:block;"></span>
      <span style="background:#00a4ef;width:9px;height:9px;display:block;"></span>
      <span style="background:#ffb900;width:9px;height:9px;display:block;"></span>
    </div>
    <span style="font-size:15px;font-weight:600;color:#323130;">MsScope — Veille Microsoft</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HERO
# ============================================================
if st.session_state.current_tab == "Accueil":
    now = datetime.now()
    date_fr = now.strftime("%A %d %B %Y")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0067b8 0%,#004578 60%,#002d6e 100%);padding:40px 32px 32px;position:relative;overflow:hidden;">
      <div style="font-size:28px;font-weight:300;color:white;margin-bottom:6px;">Bonjour, <strong>équipe INSOMEA</strong> 👋</div>
      <div style="font-size:14px;color:rgba(255,255,255,0.75);margin-bottom:24px;">{date_fr} · Voici les dernières nouveautés Microsoft</div>
      <div style="display:flex;gap:24px;">
        <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);border-radius:8px;padding:12px 20px;text-align:center;">
          <div style="font-size:24px;font-weight:600;color:white;">{len(articles)}</div>
          <div style="font-size:11px;color:rgba(255,255,255,0.7);margin-top:2px;">Articles collectés</div>
        </div>
        <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);border-radius:8px;padding:12px 20px;text-align:center;">
          <div style="font-size:24px;font-weight:600;color:white;">{len(rapport)}</div>
          <div style="font-size:11px;color:rgba(255,255,255,0.7);margin-top:2px;">Articles importants</div>
        </div>
        <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);border-radius:8px;padding:12px 20px;text-align:center;">
          <div style="font-size:24px;font-weight:600;color:white;">12</div>
          <div style="font-size:11px;color:rgba(255,255,255,0.7);margin-top:2px;">Sources actives</div>
        </div>
        <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);border-radius:8px;padding:12px 20px;text-align:center;">
          <div style="font-size:24px;font-weight:600;color:white;">Groq</div>
          <div style="font-size:11px;color:rgba(255,255,255,0.7);margin-top:2px;">Propulsé par Groq ⚡</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# NAVIGATION
# ============================================================
col1, col2, col3, col4, _ = st.columns([1, 1, 1, 1, 4])
with col1:
    if st.button("  Accueil", key="nav_accueil", type="secondary"):
        st.session_state.current_tab = "Accueil"
        st.rerun()
with col2:
    if st.button("  Articles", key="nav_articles", type="secondary"):
        st.session_state.current_tab = "Articles"
        st.rerun()
with col3:
    if st.button("  InsoBot", key="nav_insobot", type="secondary"):
        st.session_state.current_tab = "InsoBot"
        st.rerun()
with col4:
    if st.button("  Rapport Agent", key="nav_rapport", type="secondary"):
        st.session_state.current_tab = "Rapport Agent"
        st.rerun()

active_key = {"Accueil": "nav_accueil", "Articles": "nav_articles", "InsoBot": "nav_insobot", "Rapport Agent": "nav_rapport"}
st.markdown(f"""
<style>
  button[data-testid="stBaseButton-{active_key[st.session_state.current_tab]}"] {{
    color: #0067b8 !important;
    border-bottom: 3px solid #0067b8 !important;
    font-weight: 500 !important;
  }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# ACCUEIL
# ============================================================
if st.session_state.current_tab == "Accueil":
    if articles:
        article_une = articles[0]
        extrait = article_une.get("contenu", "")[:220] + "..." if len(article_une.get("contenu", "")) > 220 else article_une.get("contenu", "")
        st.markdown("""<div style="padding:20px 32px 0;"><div style="font-size:11px;font-weight:600;color:#605e5c;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:14px;">Article à la une</div></div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:white;border:1px solid #e1dfdd;border-radius:6px;display:flex;overflow:hidden;margin:0 32px 28px 32px;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
          <div style="width:320px;min-height:180px;background:linear-gradient(135deg,#0078D4,#004578);display:flex;align-items:center;justify-content:center;flex-shrink:0;position:relative;">
            <div style="position:absolute;top:12px;left:12px;background:#ffb900;color:#323130;font-size:10px;font-weight:700;padding:3px 8px;border-radius:2px;text-transform:uppercase;">⭐ À la une</div>
            <svg width="80" height="80" viewBox="0 0 23 23" fill="none" opacity="0.3">
              <path fill="#f25022" d="M0 0h11v11H0z"/><path fill="#7fba00" d="M12 0h11v11H12z"/>
              <path fill="#00a4ef" d="M0 12h11v11H0z"/><path fill="#ffb900" d="M12 12h11v11H12z"/>
            </svg>
          </div>
          <div style="padding:24px 28px;display:flex;flex-direction:column;justify-content:center;">
            <div style="font-size:11px;color:#0078D4;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">{article_une['source']}</div>
            <div style="font-size:20px;font-weight:600;color:#323130;line-height:1.3;margin-bottom:10px;">{article_une['titre']}</div>
            <div style="font-size:13px;color:#605e5c;line-height:1.6;margin-bottom:16px;">{extrait}</div>
            <a href="{article_une['lien']}" target="_blank" style="color:#0078D4;font-size:13px;font-weight:500;text-decoration:none;">Lire l'article complet →</a>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""<div style="padding:0 32px;"><div style="font-size:11px;font-weight:600;color:#605e5c;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:14px;">Actualités récentes</div></div>""", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, article in enumerate(articles[1:7]):
        with cols[i % 3]:
            date = article.get("date", "")[:10]
            extrait = article.get("contenu", "")[:130] + "..." if len(article.get("contenu", "")) > 130 else article.get("contenu", "")
            st.markdown(f"""
            <div style="background:white;border:1px solid #e1dfdd;border-radius:6px;padding:16px;margin:0 0 14px 0;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
              <div style="font-size:10px;color:#0078D4;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">{article['source']}</div>
              <div style="font-size:14px;font-weight:500;color:#323130;line-height:1.4;margin-bottom:8px;">{article['titre']}</div>
              <div style="font-size:12px;color:#605e5c;line-height:1.5;margin-bottom:12px;">{extrait}</div>
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:11px;color:#a19f9d;">📅 {date}</span>
                <a href="{article['lien']}" target="_blank" style="font-size:12px;color:#0078D4;text-decoration:none;font-weight:500;">Lire →</a>
              </div>
            </div>
            """, unsafe_allow_html=True)

    if rapport:
        st.markdown("""<div style="padding:0 32px;margin-top:8px;"><div style="font-size:11px;font-weight:600;color:#605e5c;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:14px;">Rapport Agent IA — Articles importants</div></div>""", unsafe_allow_html=True)
        cols_r = st.columns(2)
        for i, item in enumerate(rapport[:4]):
            article = item["article"]
            decision = item["decision"]
            score = decision["score"]
            border = "#107C10" if score >= 8 else "#ffb900"
            badge_style = "background:#dff6dd;color:#107C10;" if score >= 8 else "background:#fff4ce;color:#835C00;"
            with cols_r[i % 2]:
                st.markdown(f"""
                <div style="background:white;border:1px solid #e1dfdd;border-left:4px solid {border};border-radius:0 6px 6px 0;padding:16px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
                    <div style="font-size:14px;font-weight:500;color:#323130;line-height:1.4;flex:1;margin-right:10px;">{article['titre']}</div>
                    <span style="font-size:11px;font-weight:600;padding:3px 10px;border-radius:2px;white-space:nowrap;{badge_style}">{score}/10</span>
                  </div>
                  <div style="font-size:11px;color:#a19f9d;margin-bottom:8px;">{article['source']} · {article['date'][:10]}</div>
                  <div style="font-size:12px;color:#605e5c;line-height:1.5;margin-bottom:10px;">{decision['resume']}</div>
                  <a href="{article['lien']}" target="_blank" style="font-size:12px;color:#0078D4;text-decoration:none;font-weight:500;">Lire l'article →</a>
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# ARTICLES
# ============================================================
elif st.session_state.current_tab == "Articles":
    st.markdown("<div style='padding:20px 32px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;font-weight:600;color:#605e5c;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:14px;'>Tous les articles collectés</div>", unsafe_allow_html=True)
    if articles:
        for article in articles:
            date = article.get("date", "")
            extrait = article.get("contenu", "")[:180] + "..." if len(article.get("contenu", "")) > 180 else article.get("contenu", "")
            st.markdown(f"""
            <div style="background:white;border:1px solid #e1dfdd;border-radius:6px;padding:16px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
              <div style="font-size:10px;color:#0078D4;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">{article['source']}</div>
              <div style="font-size:16px;font-weight:600;color:#323130;line-height:1.4;margin-bottom:8px;">{article['titre']}</div>
              <div style="font-size:12px;color:#605e5c;line-height:1.6;margin-bottom:10px;">{extrait}</div>
              <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;">
                <span style="font-size:11px;color:#a19f9d;">📅 {date[:10] if isinstance(date, str) and len(date) >= 10 else date}</span>
                <a href="{article['lien']}" target="_blank" style="font-size:12px;color:#0078D4;text-decoration:none;font-weight:500;">Lire l'article →</a>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Aucun article collecté pour le moment.")
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# INSOBOT
# ============================================================
elif st.session_state.current_tab == "InsoBot":
    st.markdown("""
    <div style="padding:10px 32px 0;">
      <div style="background:white;border:1px solid #e1dfdd;border-radius:6px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
        <div style="background:#f9f8f7;border-bottom:1px solid #e1dfdd;padding:12px 20px;display:flex;align-items:center;gap:10px;">
          <div style="width:32px;height:32px;background:#0078D4;border-radius:50%;display:flex;align-items:center;justify-content:center;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/></svg>
          </div>
          <div>
            <div style="font-size:14px;font-weight:600;color:#323130;">InsoBot</div>
            <div style="font-size:11px;color:#107C10;">● En ligne</div>
            <div style="font-size:11px;color:#ff8c00;font-weight:600;margin-top:2px;">Propulsé par Groq ⚡</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    exemples = [
        "Mon client veut travailler à distance ?",
        "Nouveautés Azure ?",
        "Client veut sécuriser son entreprise ?",
        "Client PME veut un ERP ?",
    ]
    cols_ex = st.columns(len(exemples))
    for i, ex in enumerate(exemples):
        with cols_ex[i]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.question_exemple = ex
                st.rerun()

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    question = st.chat_input("Posez votre question à InsoBot...")
    if st.session_state.question_exemple:
        question = st.session_state.question_exemple
        st.session_state.question_exemple = None

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                status = st.status("InsoBot réfléchit…", expanded=True)
                full_response = ""
                sources_docs = []
                try:
                    q_clean = preprocess_question(question)
                    run_rag = charger_rag()
                    for chunk, docs in run_rag(q_clean):
                        if chunk:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                            status.update(label="InsoBot répond…", state="running")
                        sources_docs = docs
                    message_placeholder.markdown(full_response)
                    status.update(label="Réponse prête", state="complete")
                    if sources_docs:
                        with st.expander("📎 Sources utilisées"):
                            for i, doc in enumerate(sources_docs):
                                st.markdown(f"**{doc.metadata.get('source','')}** — {doc.metadata.get('titre','')}")
                                st.markdown(f"🔗 [Lire l'article]({doc.metadata.get('lien','')})")
                                if i < len(sources_docs) - 1:
                                    st.divider()
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    status.update(label="Erreur de génération", state="error")
                    st.error(f"Erreur : {e}")

    if st.session_state.messages:
        col_clear, _ = st.columns([1, 5])
        with col_clear:
            if st.button("🗑️ Effacer la conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

# ============================================================
# RAPPORT AGENT
# ============================================================
elif st.session_state.current_tab == "Rapport Agent":
    st.markdown("<div style='padding:20px 32px;'>", unsafe_allow_html=True)
    rapport = charger_rapport()
    if not rapport:
        if not st.session_state.get("rapport_agent_started", False):
            lancer_rapport_si_besoin()
        st.markdown("""
        <div style="background:white;border:1px solid #e1dfdd;border-left:4px solid #ffb900;border-radius:8px;padding:24px 20px;margin:10px 0 18px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
          <div style="font-size:12px;font-weight:700;color:#605e5c;text-transform:uppercase;letter-spacing:0.7px;margin-bottom:8px;">Rapport Agent</div>
          <div style="font-size:28px;font-weight:600;color:#323130;margin-bottom:8px;">Aucun article important</div>
          <div style="font-size:14px;color:#605e5c;line-height:1.6;">Aucun article ne répond actuellement aux critères d'importance commerciale pour INSOMEA.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"🔔 {len(rapport)} article(s) important(s) détecté(s)")
        cols_r = st.columns(2)
        for i, item in enumerate(rapport):
            article = item["article"]
            decision = item["decision"]
            score = decision["score"]
            border = "#107C10" if score >= 8 else "#ffb900"
            badge_style = "background:#dff6dd;color:#107C10;" if score >= 8 else "background:#fff4ce;color:#835C00;"
            with cols_r[i % 2]:
                st.markdown(f"""
                <div style="background:white;border:1px solid #e1dfdd;border-left:4px solid {border};border-radius:0 6px 6px 0;padding:16px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
                    <div style="font-size:14px;font-weight:500;color:#323130;line-height:1.4;flex:1;margin-right:10px;">{article['titre']}</div>
                    <span style="font-size:11px;font-weight:600;padding:3px 10px;border-radius:2px;white-space:nowrap;{badge_style}">{score}/10</span>
                  </div>
                  <div style="font-size:11px;color:#a19f9d;margin-bottom:8px;">{article['source']} · {article['date'][:10]}</div>
                  <div style="font-size:12px;color:#605e5c;line-height:1.5;margin-bottom:10px;">
                    <b>Pourquoi :</b> {decision['raison']}<br>
                    <b>Résumé :</b> {decision['resume']}
                  </div>
                  <a href="{article['lien']}" target="_blank" style="font-size:12px;color:#0078D4;text-decoration:none;font-weight:500;">Lire l'article →</a>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)