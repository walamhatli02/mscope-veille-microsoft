import streamlit as st
import json
import os
import html
import hashlib
import unicodedata
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq_utils import SelecteurModele, erreur_modele, erreur_quota
from scraper import SOURCES

load_dotenv()

# Score de pertinence minimal d'un extrait (0 = filtre seulement les extraits sans rapport).
# A CALIBRER avec les scores affichés dans « Sources utilisées ».
SEUIL_PERTINENCE = float(os.getenv("SEUIL_PERTINENCE", "0.0"))
RAPPORT_PATH = "data/rapport_agent.json"
ARTICLES_PATH = "data/articles.json"
ENVOYES_PATH = "data/articles_envoyes.json"
INSOMEA_PATH = "insomea_info.txt"

st.set_page_config(page_title="MsScope — Veille Microsoft", page_icon="🔎", layout="wide")

PAGES = [("Accueil", "nav_accueil"), ("Articles", "nav_articles"),
         ("InsoBot", "nav_insobot"), ("Rapport de l'agent", "nav_rapport")]
for cle, defaut in [("page", "Accueil"), ("messages", []), ("question_exemple", None), ("nb_aff", 30)]:
    if cle not in st.session_state:
        st.session_state[cle] = defaut

# ============================================================
# THÈME (bleu Microsoft dominant, 4 couleurs seulement dans la barre du haut)
# ============================================================
st.html("""<style>
.stApp{background:#F3F2F1;color:#201F1E}
.stApp,.stApp button,.stApp input,.stApp textarea{font-family:'Segoe UI','Segoe UI Variable',system-ui,-apple-system,sans-serif}
header[data-testid="stHeader"],section[data-testid="stSidebar"]{display:none!important}
[data-testid="stMainBlockContainer"],.block-container{padding:0!important;max-width:100%!important}
.ms-barre{display:grid;grid-template-columns:repeat(4,1fr);height:4px}
.st-key-entete{background:#FFFFFF;border-bottom:1px solid #EDEBE9}
.st-key-entete_ligne{padding:0 max(24px,calc((100% - 1232px)/2));min-height:54px;flex-wrap:wrap;column-gap:4px}
.st-key-entete_ligne .stButton button{border-radius:0;border-bottom:3px solid transparent;padding:14px 12px 11px;color:#323130;white-space:nowrap}
.st-key-entete_ligne .stButton button:hover,.st-key-entete_ligne .stButton button:focus{color:#0067B8;background:transparent!important}
.ms-marque{display:flex;align-items:center;gap:10px;margin-right:16px;font-family:'Segoe UI','Segoe UI Variable',system-ui,-apple-system,sans-serif;color:#201F1E}
.ms-logo{width:26px;height:26px;flex:none;background:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20viewBox%3D%220%200%2028%2028%22%3E%3Crect%20width%3D%2228%22%20height%3D%2228%22%20rx%3D%224%22%20fill%3D%22%230078D4%22/%3E%3Ccircle%20cx%3D%2213%22%20cy%3D%2213%22%20r%3D%226.5%22%20fill%3D%22none%22%20stroke%3D%22%23FFFFFF%22%20stroke-width%3D%222.2%22/%3E%3Cpath%20d%3D%22M18%2018l5%205%22%20stroke%3D%22%23FFFFFF%22%20stroke-width%3D%222.4%22%20stroke-linecap%3D%22round%22/%3E%3Ccircle%20cx%3D%2213%22%20cy%3D%2213%22%20r%3D%222%22%20fill%3D%22%23FFB900%22/%3E%3C/svg%3E") center/26px no-repeat}
.ms-marque-nom{font-size:20px;font-weight:600;letter-spacing:-.3px}
.ms-marque-sep{width:1px;height:22px;background:#C8C6C4}
.ms-gris{color:#605E5C;font-size:13px}
.st-key-hero{background:#0078D4;padding:56px max(24px,calc((100% - 1232px)/2))}
.ms-bandeau{background:#0078D4;color:#FFFFFF;padding:32px max(24px,calc((100% - 1232px)/2))}
.ms-blanc{color:#FFFFFF!important}
h1.ms-h1{font-size:44px!important;font-weight:600!important;line-height:1.15!important;margin:10px 0 12px!important;padding:0!important;color:#FFFFFF!important;max-width:820px}
h1.ms-h1-page{font-size:34px!important;font-weight:600!important;margin:0 0 6px!important;padding:0!important;color:#FFFFFF!important}
h2.ms-h2{font-size:22px!important;font-weight:600!important;margin:0!important;padding:0!important;color:#201F1E!important}
.st-key-btn_prios button{background:#FFFFFF!important;color:#0067B8!important;border:1px solid #FFFFFF!important;border-radius:2px!important;font-weight:600!important;padding:10px 22px!important}
.st-key-btn_bot button{background:transparent!important;color:#FFFFFF!important;border:1px solid #FFFFFF!important;border-radius:2px!important;font-weight:600!important;padding:10px 22px!important}
.st-key-hero_boutons{margin-top:12px}
.st-key-contenu{padding:32px max(24px,calc((100% - 1232px)/2)) 56px}
.ms-grille{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
.ms-carte{background:#FFFFFF;padding:20px;display:flex;flex-direction:column;gap:10px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.ms-kpi{background:#FFFFFF;padding:18px 20px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.ms-kpi-val{font-size:32px;font-weight:600;display:block;line-height:1.3;word-break:break-word}
.ms-titre{font-size:16px;font-weight:600;line-height:1.35}
.ms-pastille{color:#FFFFFF;font-size:12px;font-weight:600;padding:2px 9px;border-radius:2px;white-space:nowrap}
.ms-score{font-size:13px;font-weight:700;color:#0067B8;background:#DEECF9;padding:2px 8px;border-radius:2px;white-space:nowrap}
.ms-statut{font-size:12px;color:#323130;background:#EDEBE9;padding:2px 8px;border-radius:2px;white-space:nowrap}
.ms-ligne-meta{display:flex;flex-wrap:wrap;align-items:center;gap:8px}
.ms-lien{color:#0067B8!important;font-weight:600;font-size:14px;text-decoration:none!important}
h3.ms-h3{font-size:16px!important;font-weight:600!important;margin:0 0 4px!important;padding:0!important;color:#201F1E!important}
.ms-lien:hover{color:#005A9E!important;text-decoration:underline}
.ms-liste{background:#FFFFFF;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.ms-rangee{display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px;padding:14px 16px;border-bottom:1px solid #F3F2F1}
.ms-rangee-txt{flex:999 1 360px;min-width:0;display:flex;flex-direction:column;gap:4px}
.ms-label{font-size:12px;font-weight:700;letter-spacing:.3px}
.ms-barre-score{flex:1;height:10px;background:#EDEBE9}
.st-key-filtres,.st-key-aside,.st-key-chat{background:#FFFFFF;padding:16px 20px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.ms-pied{background:#FFFFFF;border-top:1px solid #EDEBE9;padding:18px max(24px,calc((100% - 1232px)/2));display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;font-size:13px;color:#605E5C}
</style>""")

# ============================================================
# DONNÉES ET UTILITAIRES
# ============================================================
def charger_json(chemin, defaut):
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, type(defaut)) else defaut
    except (OSError, ValueError):
        return defaut

def e(texte):
    """Échappe un texte venant d'un flux RSS ou du LLM avant de l'injecter dans du HTML."""
    return html.escape(str(texte or ""), quote=True)

def url_sure(url):
    url = str(url or "")
    return html.escape(url, quote=True) if url.startswith(("https://", "http://")) else "#"

FAMILLES = {
    "Azure Updates": ("Azure", "#0078D4"), "Azure AI Blog": ("Azure", "#0078D4"),
    "Microsoft 365 Blog": ("Microsoft 365", "#D83B01"), "Microsoft Teams Blog": ("Microsoft 365", "#D83B01"),
    "Microsoft Security Blog": ("Sécurité", "#107C10"),
    "Microsoft Dynamics 365 Blog": ("Dynamics 365", "#5C2D91"),
    "Microsoft Power Platform Blog": ("Power Platform", "#5C2D91"),
    "Windows Blog": ("Windows & Surface", "#0067B8"),
}

def pastille(source):
    nom, couleur = FAMILLES.get(source, ("Microsoft", "#505050"))
    return f'<span class="ms-pastille" style="background:{couleur}">{e(nom)}</span>'

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
        "septembre", "octobre", "novembre", "décembre"]

def maintenant():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Africa/Tunis"))
    except Exception:          # Windows sans le paquet tzdata
        return datetime.now()

def date_longue_fr(dt):
    return f"{JOURS[dt.weekday()].capitalize()} {dt.day} {MOIS[dt.month - 1]} {dt.year}"

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
@st.cache_resource(show_spinner=False)
def selecteur_chat():
    # Partagé entre sessions : un modèle en 404 n'est essayé qu'une fois.
    return SelecteurModele("GROQ_MODEL_CHAT", max_tokens=1024)

def date_modif(chemin):
    return os.path.getmtime(chemin) if os.path.exists(chemin) else 0

def id_chunk(*parties):
    return hashlib.md5("|".join(str(p) for p in parties).encode("utf-8")).hexdigest()

@st.cache_resource(max_entries=1, show_spinner="Préparation de la base d'articles (première question uniquement)…")
def charger_rag(_version_articles):
    """Réindexe quand articles.json change ; les ids fixes évitent les doublons."""
    import chromadb
    from langchain_chroma import Chroma

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    client = chromadb.EphemeralClient()

    vs_microsoft = Chroma(client=client, collection_name="microsoft_articles", embedding_function=embeddings)
    vs_insomea = Chroma(client=client, collection_name="insomea_info", embedding_function=embeddings)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    if os.path.exists(ARTICLES_PATH):
        with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
            articles = json.load(f)
        docs, ids, vus = [], [], set()
        for article in articles:
            lien = article.get("lien", "")
            if lien in vus:
                continue   # même article publié par deux sources
            vus.add(lien)
            chunks = splitter.split_text(f"{article['titre']}\n\n{article.get('contenu', '')}")
            for i, chunk in enumerate(chunks):
                docs.append(Document(page_content=chunk, metadata={
                    "source": article.get("source", ""),
                    "titre": article.get("titre", ""),
                    "lien": article.get("lien", ""),
                    "date": article.get("date", ""),
                    "chunk": i
                }))
                ids.append(id_chunk(article.get("lien", ""), i))
        if docs:
            vs_microsoft.add_documents(docs, ids=ids)

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
            vs_insomea.add_documents(docs_ins, ids=[f"insomea-{i}" for i in range(len(docs_ins))])

    def chercher(vs, question, k):
        gardes = []
        for doc, score in vs.similarity_search_with_relevance_scores(question, k=k):
            if score >= SEUIL_PERTINENCE:
                doc.metadata["score"] = round(score, 2)
                gardes.append(doc)
        return gardes

    prompt = ChatPromptTemplate.from_template("""Tu es InsoBot, assistant Microsoft pour l'équipe commerciale d'INSOMEA en Tunisie.
Réponds TOUJOURS en français, même si la question est mal formulée.
Appuie-toi UNIQUEMENT sur les informations INSOMEA et les actualités Microsoft ci-dessous.
Si elles ne permettent pas de répondre, dis clairement « Je n'ai pas trouvé cette information dans la veille MsScope. » au lieu d'inventer.
Réponds de façon claire et concise (2 à 5 phrases).

Informations INSOMEA: {context_insomea}
Actualités Microsoft: {context_microsoft}
Question: {question}
Réponse en français:""")

    def formater(docs):
        return "\n\n".join(f"[{doc.metadata.get('titre','')}]\n{doc.page_content}" for doc in docs)

    def run_rag(question):
        selecteur = selecteur_chat()
        try:
            docs_microsoft = chercher(vs_microsoft, question, k=4)
            docs_insomea = chercher(vs_insomea, question, k=3)
        except Exception:
            docs_microsoft = []
            docs_insomea = []

        payload = {
            "context_microsoft": formater(docs_microsoft) if docs_microsoft else "Pas d'actualités disponibles.",
            "context_insomea": formater(docs_insomea) if docs_insomea else "Pas d'infos INSOMEA.",
            "question": question
        }

        sources = docs_insomea + docs_microsoft   # exactement ce que le modèle a reçu
        # Vrai streaming ; repli de modèle seulement tant qu'aucun mot n'a été affiché.
        while True:
            chain = prompt | selecteur.llm(streaming=True) | StrOutputParser()
            deja_envoye = False
            try:
                for morceau in chain.stream(payload):
                    deja_envoye = True
                    yield morceau, sources
                return
            except Exception as exc:
                if not deja_envoye and erreur_modele(exc):
                    selecteur.index += 1
                    continue
                if not deja_envoye and erreur_quota(exc):
                    yield "Limite de requêtes Groq atteinte. Réessayez dans une minute.", sources
                    return
                raise

    return run_rag

def date_courte(date_str):
    dt = parse_date_article(date_str)
    return dt.strftime("%d/%m/%Y") if dt != datetime.min else "date inconnue"

# ============================================================
# CHARGEMENT (lecture seule : la collecte et l'agent tournent dans GitHub Actions)
# ============================================================
articles = sorted(charger_json(ARTICLES_PATH, []), key=lambda a: parse_date_article(a.get("date", "")), reverse=True)
rapport = [it for it in charger_json(RAPPORT_PATH, []) if isinstance(it, dict) and "article" in it and "decision" in it]
rapport.sort(key=lambda it: (it["decision"].get("score", 0), parse_date_article(it["article"].get("date", ""))), reverse=True)
liens_analyses = {a.get("lien") for a in charger_json(ENVOYES_PATH, [])}
scores = {it["article"].get("lien"): it["decision"].get("score", 0) for it in rapport}
sources_actives = sorted({a.get("source") for a in articles if a.get("source")})
nb_en_attente = sum(1 for a in articles if a.get("lien") not in liens_analyses)

AVATARS = {"user": ":material/person:", "assistant": ":material/smart_toy:"}

def naviguer(page):
    st.session_state.page = page

def demander(question):
    st.session_state.question_exemple = question
    st.session_state.page = "InsoBot"

def reinitialiser_pagination():
    st.session_state.nb_aff = 30

def bandeau(titre, sous_titre):
    st.markdown(f'<div class="ms-bandeau"><h1 class="ms-h1-page">{e(titre)}</h1>'
                f'<div class="ms-blanc" style="font-size:15px">{e(sous_titre)}</div></div>',
                unsafe_allow_html=True, anchors=False)

def entete_section(titre, bouton=None, cible=None, cle=None):
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown(f'<h2 class="ms-h2">{e(titre)}</h2>', unsafe_allow_html=True, anchors=False, width="content")
        if bouton:
            with st.container(key=f"{cle}_zone", horizontal=True, horizontal_alignment="right"):
                st.button(bouton, key=cle, type="tertiary", on_click=naviguer, args=(cible,))

def badge_article(lien):
    if lien in scores:
        return f'<span class="ms-score">{scores[lien]}/10</span>'
    if lien in liens_analyses:
        return '<span class="ms-statut">Non prioritaire</span>'
    return '<span class="ms-statut">Non analysé</span>'

def carte_priorite(item, details=False):
    a, d = item["article"], item["decision"]
    corps = ""
    if details:
        corps = (f'<div><div class="ms-label" style="color:#0067B8">POURQUOI C\'EST IMPORTANT</div>'
                 f'<div style="font-size:14px;line-height:1.55">{e(d.get("raison"))}</div></div>'
                 f'<div><div class="ms-label" style="color:#605E5C">RÉSUMÉ</div>'
                 f'<div style="font-size:14px;line-height:1.55">{e(d.get("resume"))}</div></div>')
    return (f'<div class="ms-carte"><div class="ms-ligne-meta">{pastille(a.get("source"))}'
            f'<span style="margin-left:auto" class="ms-score">{e(d.get("score", "?"))}/10</span></div>'
            f'<div class="ms-titre">{e(a.get("titre"))}</div>'
            f'<div class="ms-gris">{e(a.get("source"))} · {date_courte(a.get("date"))}</div>{corps}'
            f'<a class="ms-lien" href="{url_sure(a.get("lien"))}" target="_blank" rel="noopener">Lire l\'article ›</a></div>')

def carte_article(a):
    return (f'<div class="ms-carte"><div class="ms-ligne-meta">{pastille(a.get("source"))}'
            f'<span style="margin-left:auto">{badge_article(a.get("lien"))}</span></div>'
            f'<div class="ms-titre">{e(a.get("titre"))}</div>'
            f'<div class="ms-gris">{e(a.get("source"))} · {date_courte(a.get("date"))}</div>'
            f'<a class="ms-lien" href="{url_sure(a.get("lien"))}" target="_blank" rel="noopener">Lire l\'article ›</a></div>')

def grille(cartes):
    st.markdown('<div class="ms-grille">' + "".join(cartes) + '</div>', unsafe_allow_html=True)

def kpis(tuiles):
    st.markdown('<div class="ms-grille" style="grid-template-columns:repeat(auto-fit,minmax(210px,1fr))">' + "".join(
        f'<div class="ms-kpi"><span class="ms-kpi-val" style="{style}">{e(val)}</span>'
        f'<span class="ms-gris" style="font-size:14px">{e(label)}</span></div>'
        for val, label, style in tuiles) + '</div>', unsafe_allow_html=True)

# ============================================================
# PAGE
# ============================================================
with st.container(key="page", gap=None):

    # ---------- En-tête et navigation
    cle_active = dict(PAGES)[st.session_state.page]
    st.html(f"<style>.st-key-{cle_active} button{{border-bottom-color:#0078D4!important;"
            f"font-weight:600!important;color:#201F1E!important}}</style>")
    with st.container(key="entete", gap=None):
        st.markdown('<div class="ms-barre"><div style="background:#F25022"></div><div style="background:#7FBA00">'
                    '</div><div style="background:#00A4EF"></div><div style="background:#FFB900"></div></div>',
                    unsafe_allow_html=True)
        with st.container(key="entete_ligne", horizontal=True, vertical_alignment="center", gap=None):
            st.html('<div class="ms-marque"><span class="ms-logo" aria-hidden="true"></span>'
                    '<span class="ms-marque-nom">MsScope</span><span class="ms-marque-sep"></span>'
                    '<span class="ms-gris" style="font-size:14px">Veille Microsoft</span></div>',
                    width="content")
            for nom, cle in PAGES:
                st.button(nom, key=cle, type="tertiary", on_click=naviguer, args=(nom,))
            with st.container(key="entete_droite", horizontal=True, horizontal_alignment="right"):
                st.markdown('<span class="ms-gris">INSOMEA · Partenaire Microsoft</span>', unsafe_allow_html=True)

    page = st.session_state.page

    # ---------- ACCUEIL
    if page == "Accueil":
        with st.container(key="hero"):
            st.markdown(f'<div class="ms-blanc" style="font-size:14px;font-weight:600">{e(date_longue_fr(maintenant()))}</div>'
                        '<h1 class="ms-h1">Les nouveautés Microsoft qui comptent pour vos clients</h1>'
                        '<div class="ms-blanc" style="font-size:16px;max-width:820px">Collectées chaque jour, triées par '
                        "l'agent IA, prêtes pour l'équipe commerciale d'INSOMEA.</div>",
                        unsafe_allow_html=True, anchors=False)
            with st.container(horizontal=True, gap="small", key="hero_boutons"):
                st.button(f"Voir les {len(rapport)} priorités", key="btn_prios", on_click=naviguer, args=("Rapport de l'agent",))
                st.button("Poser une question à InsoBot", key="btn_bot", on_click=naviguer, args=("InsoBot",))

        with st.container(key="contenu", gap="large"):
            kpis([(len(articles), "Articles collectés", ""),
                  (len(rapport), "Jugés importants par l'agent", ""),
                  (len(sources_actives), f"Sources actives sur {len(SOURCES)} suivies", ""),
                  (selecteur_chat().modele or "aucun", "Modèle d'analyse · Groq", "font-size:22px;padding:6px 0 4px")])

            with st.container(gap="small"):
                entete_section("Priorités commerciales", "Voir tout le rapport ›", "Rapport de l'agent", "lien_rapport")
                if rapport:
                    grille([carte_priorite(it) for it in rapport[:4]])
                else:
                    st.info("Aucun article important pour le moment : l'agent n'a encore rien retenu.")

            with st.container(gap="small"):
                entete_section("Dernières publications", "Tous les articles ›", "Articles", "lien_articles")
                if articles:
                    grille([carte_article(a) for a in articles[:3]])
                else:
                    st.info("Aucun article collecté pour le moment.")

    # ---------- ARTICLES
    elif page == "Articles":
        bandeau("Articles collectés", f"{len(articles)} articles · {len(sources_actives)} sources actives")
        with st.container(key="contenu", gap="medium"):
            with st.container(key="filtres", gap="small"):
                with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
                    q = st.text_input("Rechercher", placeholder="Copilot, Defender, Dynamics…", key="f_q",
                                      on_change=reinitialiser_pagination)
                    tri = st.selectbox("Trier par", ["Plus récents", "Meilleur score"], key="f_tri", width=220,
                                       on_change=reinitialiser_pagination)
                    seulement = st.toggle("Importants seulement (score ≥ 6)", key="f_imp",
                                          on_change=reinitialiser_pagination)
                source = st.pills("Source", ["Toutes"] + sources_actives, default="Toutes", required=True,
                                  key="f_src", on_change=reinitialiser_pagination)

            terme = (q or "").lower().strip()
            res = [a for a in articles
                   if (not terme or terme in a.get("titre", "").lower() or terme in a.get("contenu", "").lower())
                   and (source in (None, "Toutes") or a.get("source") == source)
                   and (not seulement or a.get("lien") in scores)]
            if tri == "Meilleur score":
                res.sort(key=lambda a: scores.get(a.get("lien"), -1), reverse=True)

            nb = st.session_state.nb_aff
            st.markdown(f'<div class="ms-gris">{len(res)} article(s) trouvé(s) · {min(nb, len(res))} affiché(s)</div>',
                        unsafe_allow_html=True)
            if res:
                st.markdown('<div class="ms-liste">' + "".join(
                    f'<div class="ms-rangee">{badge_article(a.get("lien"))}<div class="ms-rangee-txt">'
                    f'<span class="ms-titre" style="font-size:15px">{e(a.get("titre"))}</span>'
                    f'<span class="ms-ligne-meta ms-gris" style="font-size:12px">{pastille(a.get("source"))}'
                    f'{e(a.get("source"))} · {date_courte(a.get("date"))}</span></div>'
                    f'<a class="ms-lien" href="{url_sure(a.get("lien"))}" target="_blank" rel="noopener">Lire l\'article ›</a></div>'
                    for a in res[:nb]) + '</div>', unsafe_allow_html=True)
                if len(res) > nb:
                    st.button(f"Afficher {min(30, len(res) - nb)} articles de plus", key="plus",
                              on_click=lambda: st.session_state.update(nb_aff=st.session_state.nb_aff + 30))
            else:
                st.info("Aucun article ne correspond. Modifiez la recherche ou les filtres.")

    # ---------- INSOBOT
    elif page == "InsoBot":
        bandeau("InsoBot", "L'assistant de l'équipe commerciale : il répond à partir de la veille MsScope et cite ses sources.")
        with st.container(key="contenu"):
            col_chat, col_aside = st.columns([2.3, 1], gap="large")
            with col_aside:
                with st.container(key="aside"):
                    st.markdown('<h3 class="ms-h3">Questions fréquentes</h3>',
                                unsafe_allow_html=True, anchors=False)
                    for i, ex in enumerate(["Mon client veut travailler à distance ?", "Nouveautés Azure ?",
                                            "Mon client veut sécuriser son entreprise ?", "Une PME cherche un ERP ?"]):
                        st.button(ex, key=f"ex_{i}", width="stretch", on_click=demander, args=(ex,))
                    st.markdown('<h3 class="ms-h3" style="margin-top:12px!important">'
                                'Comment InsoBot répond</h3><div style="font-size:13px;line-height:1.55;color:#323130">'
                                "Uniquement à partir des articles collectés et de la fiche INSOMEA. Chaque réponse cite "
                                "ses sources ; si l'information manque, InsoBot le dit au lieu d'inventer.</div>"
                                f'<div class="ms-gris" style="margin-top:6px">Modèle : {e(selecteur_chat().modele)} via Groq</div>',
                                unsafe_allow_html=True, anchors=False)

            with col_chat:
                with st.container(key="chat"):
                    zone_messages = st.container()
                    zone_effacer = st.container()
                    zone_accueil = st.empty()
                    with zone_messages:
                        if not st.session_state.messages:
                            zone_accueil.markdown('<div class="ms-gris" style="font-size:14px">Posez une question ou choisissez '
                                        'une question fréquente à droite.</div>', unsafe_allow_html=True)
                        for msg in st.session_state.messages:
                            with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
                                st.markdown(msg["content"])

                    question = st.chat_input("Posez votre question à InsoBot…")
                    if st.session_state.question_exemple:
                        question = st.session_state.question_exemple
                        st.session_state.question_exemple = None

                    if question:
                        zone_accueil.empty()
                        st.session_state.messages.append({"role": "user", "content": question})
                        with zone_messages, st.chat_message("user", avatar=AVATARS["user"]):
                            st.markdown(question)
                        with zone_messages, st.chat_message("assistant", avatar=AVATARS["assistant"]):
                            zone = st.empty()
                            reponse, sources_docs = "", []
                            try:
                                run_rag = charger_rag(date_modif(ARTICLES_PATH))
                                for morceau, docs in run_rag(preprocess_question(question)):
                                    reponse += morceau
                                    sources_docs = docs
                                    zone.markdown(reponse + "▌")
                                zone.markdown(reponse)
                                if sources_docs:
                                    with st.expander("📎 Sources utilisées"):
                                        for doc in sources_docs:
                                            m = doc.metadata
                                            st.markdown(f"**{m.get('source', '')}** — [{m.get('titre', '')}]({m.get('lien', '')}) "
                                                        f"· pertinence {m.get('score', '?')}")
                                st.session_state.messages.append({"role": "assistant", "content": reponse})
                            except Exception as exc:
                                st.error(f"Erreur : {exc}")

                    if st.session_state.messages:
                        zone_effacer.button("🗑️ Effacer la conversation", key="effacer",
                                  on_click=lambda: st.session_state.update(messages=[]))

    # ---------- RAPPORT DE L'AGENT
    else:
        bandeau("Rapport de l'agent", "Les articles que l'agent IA juge importants pour la vente (score de 6 à 10), "
                                      "avec la raison et un résumé en français.")
        with st.container(key="contenu", gap="large"):
            compte = {s: sum(1 for it in rapport if it["decision"].get("score") == s) for s in range(10, 5, -1)}
            maxi = max(compte.values()) or 1
            barres = "".join(
                f'<div style="display:flex;align-items:center;gap:10px;font-size:13px;margin-top:6px">'
                f'<span style="width:44px">{s}/10</span><div class="ms-barre-score"><div style="height:10px;'
                f'background:#0078D4;width:{round(n / maxi * 100)}%"></div></div>'
                f'<span style="width:28px;text-align:right;font-weight:600">{n}</span></div>'
                for s, n in compte.items())
            st.markdown(
                '<div class="ms-grille" style="grid-template-columns:repeat(auto-fit,minmax(260px,1fr))">'
                f'<div class="ms-kpi"><span class="ms-kpi-val">{len(rapport)}</span>'
                '<span class="ms-gris" style="font-size:14px">Articles importants dans le rapport</span></div>'
                f'<div class="ms-kpi"><div style="font-size:14px;font-weight:600">Répartition des scores</div>{barres}</div>'
                f'<div class="ms-kpi"><div style="font-size:14px;font-weight:600">Avancement de l\'analyse</div>'
                f'<div style="font-size:13px;line-height:1.7;margin-top:6px">{len(articles) - nb_en_attente} articles analysés'
                f'<br>{nb_en_attente} en attente d\'analyse</div>'
                '<div class="ms-gris" style="margin-top:4px">L\'agent tourne chaque jour via GitHub Actions.</div></div>'
                '</div>', unsafe_allow_html=True)

            with st.container(gap="small"):
              with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown('<h2 class="ms-h2">Articles du rapport</h2>', unsafe_allow_html=True, anchors=False, width="content")
                seuils = {6: f"Tous ({len(rapport)})",
                          8: f"8 et plus ({sum(1 for it in rapport if it['decision'].get('score', 0) >= 8)})",
                          9: f"9 et plus ({sum(1 for it in rapport if it['decision'].get('score', 0) >= 9)})"}
                with st.container(key="filtre_score_zone", horizontal=True, horizontal_alignment="right"):
                    mini = st.segmented_control("Score minimum", list(seuils), format_func=seuils.get, default=6,
                                                required=True, key="f_score", label_visibility="collapsed")
              choisis = [it for it in rapport if it["decision"].get("score", 0) >= (mini or 6)]
              if choisis:
                st.markdown('<div class="ms-grille" style="grid-template-columns:repeat(auto-fit,minmax(360px,1fr))">'
                            + "".join(carte_priorite(it, details=True) for it in choisis) + '</div>',
                            unsafe_allow_html=True)
              else:
                st.info("Aucun article important pour le moment.")

    # ---------- Pied de page
    dernier = date_courte(articles[0].get("date")) if articles else "—"
    st.markdown(f'<div class="ms-pied"><span>MsScope · outil interne INSOMEA</span>'
                f'<span>Dernier article publié : {dernier} · sources : flux RSS officiels Microsoft</span></div>',
                unsafe_allow_html=True)