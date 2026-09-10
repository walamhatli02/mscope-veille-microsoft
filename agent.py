import json
import os
import sys
from dotenv import load_dotenv
# Fix UnicodeEncodeError on Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
def get_groq_model():
    preferred = os.getenv("GROQ_MODEL")
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend([
        "groq/compound",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "groq/compound-mini",
    ])
    for model in candidates:
        if model:
            return model
    return "groq/compound"

LLM_MODEL = get_groq_model()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ARTICLES_PATH = "data/articles.json"
ARTICLES_ENVOYES_PATH = "data/articles_envoyes.json"
RAPPORT_PATH = "data/rapport_agent.json"

# ============================================================
# ÉTAPE 1 : Charger les articles non encore traités
# ============================================================
def charger_nouveaux_articles(force=False):
    print("📂 Chargement des articles...")

    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        tous = json.load(f)

    envoyes = []
    if os.path.exists(ARTICLES_ENVOYES_PATH):
        with open(ARTICLES_ENVOYES_PATH, "r", encoding="utf-8") as f:
            envoyes = json.load(f)

    liens_envoyes = {a["lien"] for a in envoyes}
    nouveaux = [a for a in tous if a["lien"] not in liens_envoyes]

    if not nouveaux and (force or not os.path.exists(RAPPORT_PATH)):
        print("  ⚠️ Aucun article nouveau détecté, mais un rapport est manquant : analyse complète des articles.")
        return tous, []

    print(f"  ✅ {len(nouveaux)} nouveaux articles à analyser")
    return nouveaux, envoyes

# ============================================================
# ÉTAPE 2 : L'agent décide si l'article est important
# ============================================================
def evaluer_importance(article, llm):
    prompt = ChatPromptTemplate.from_template("""
Tu es un agent IA spécialisé dans les technologies Microsoft pour une équipe commerciale qui vend des licences Microsoft.

Analyse cet article et décide s'il est IMPORTANT pour l'équipe commerciale d'INSOMEA.

Un article est IMPORTANT si :
- Il annonce une nouvelle licence ou un changement de prix
- Il présente une nouvelle fonctionnalité majeure de Microsoft 365, Azure, Dynamics ou Power Platform
- Il concerne Copilot ou l'IA Microsoft (très vendeur en ce moment)
- Il annonce un nouveau produit ou service Microsoft
- Il peut aider l'équipe à mieux vendre ou conseiller les clients

Un article est PAS IMPORTANT si :
- C'est un article générique ou trop technique
- Il ne concerne pas directement les produits vendus par INSOMEA
- Il ne présente pas de nouveauté significative

ARTICLE :
Titre : {titre}
Source : {source}
Contenu : {contenu}

Réponds UNIQUEMENT avec ce format JSON (sans texte avant ou après) :
{{
  "important": true ou false,
  "score": nombre entre 0 et 10,
  "raison": "explication courte en français",
  "resume": "résumé de l'article en 2-3 phrases en français pour l'équipe"
}}
""")

    chain = prompt | llm | StrOutputParser()

    contenu = (article.get("contenu") or "")
    contenu = " ".join(contenu.split())
    contenu = contenu[:1200]

    try:
        resultat = chain.invoke({
            "titre": article["titre"],
            "source": article["source"],
            "contenu": contenu
        })
    except Exception:
        return {
            "important": False,
            "score": 0,
            "raison": "Erreur de génération ou contenu trop volumineux",
            "resume": ""
        }

    try:
        resultat_clean = resultat.strip()
        if "```" in resultat_clean:
            resultat_clean = resultat_clean.split("```")[1]
            if resultat_clean.startswith("json"):
                resultat_clean = resultat_clean[4:]
        decision = json.loads(resultat_clean)
    except:
        decision = {
            "important": False,
            "score": 0,
            "raison": "Erreur d'analyse",
            "resume": ""
        }

    return decision

# ============================================================
# ÉTAPE 3 : Afficher le rapport dans le terminal
# ============================================================
def afficher_rapport(articles_importants):
    print("\n" + "=" * 60)
    print("📊 RAPPORT DE VEILLE MICROSOFT")
    print("=" * 60)

    if not articles_importants:
        print("  📭 Aucun article important détecté.")
        return

    print(f"  🔔 {len(articles_importants)} articles importants détectés :\n")

    for i, item in enumerate(articles_importants):
        article = item["article"]
        decision = item["decision"]
        print(f"  ⭐ [{i+1}] {article['titre']}")
        print(f"       Source  : {article['source']}")
        print(f"       Score   : {decision['score']}/10")
        print(f"       Raison  : {decision['raison']}")
        print(f"       Résumé  : {decision['resume']}")
        print(f"       Lien    : {article['lien']}")
        print()

# ============================================================
# ÉTAPE 4 : Sauvegarder le rapport en JSON
# ============================================================
def sauvegarder_rapport(articles_importants):
    os.makedirs("data", exist_ok=True)
    with open(RAPPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(articles_importants, f, ensure_ascii=False, indent=2)
    print(f"  💾 Rapport sauvegardé dans {RAPPORT_PATH}")

def sauvegarder_articles_traites(nouveaux, envoyes):
    tous_envoyes = envoyes + nouveaux
    with open(ARTICLES_ENVOYES_PATH, "w", encoding="utf-8") as f:
        json.dump(tous_envoyes, f, ensure_ascii=False, indent=2)

# ============================================================
# FONCTION PRINCIPALE
# ============================================================
def lancer_agent(force=False):
    print("=" * 60)
    print("🤖 Démarrage de l'Agent IA de veille Microsoft")
    print("=" * 60)

    nouveaux_articles, envoyes = charger_nouveaux_articles(force=force)

    if not nouveaux_articles:
        if force or not os.path.exists(RAPPORT_PATH):
            sauvegarder_rapport([])
        print("✅ Aucun article à analyser.")
        return

    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.2, max_tokens=300)

    articles_importants = []
    print("\n🔍 Analyse des articles en cours...\n")

    for i, article in enumerate(nouveaux_articles):
        print(f"  [{i+1}/{len(nouveaux_articles)}] {article['titre'][:60]}...")
        try:
            decision = evaluer_importance(article, llm)
        except Exception as e:
            print(f"    → ⚠️ Erreur lors de l'analyse : {e}")
            decision = {
                "important": False,
                "score": 0,
                "raison": "Erreur d'analyse",
                "resume": ""
            }

        statut = "⭐ IMPORTANT" if decision["important"] else "⏭️  ignoré"
        print(f"    → {statut} (score: {decision['score']}/10) — {decision['raison'][:60]}")

        if decision["important"] and decision["score"] >= 6:
            articles_importants.append({
                "article": article,
                "decision": decision
            })

    afficher_rapport(articles_importants)
    sauvegarder_rapport(articles_importants)
    sauvegarder_articles_traites(nouveaux_articles, envoyes)

    print("=" * 60)
    print("✅ Agent terminé !")
    print("=" * 60)


if __name__ == "__main__":
    lancer_agent()