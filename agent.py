import json
import os
import re
import sys
import time
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from groq_utils import SelecteurModele

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
ARTICLES_PATH = "data/articles.json"
ARTICLES_ENVOYES_PATH = "data/articles_envoyes.json"
RAPPORT_PATH = "data/rapport_agent.json"
SEUIL_SCORE = 6
MAX_RAPPORT = None        # pas de plafond : tous les articles importants restent dans le rapport
PAUSE_ENTRE_APPELS = 8    # secondes ; limite gratuite observée : 8000 tokens/min sur gpt-oss-20b

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
def evaluer_importance(article, selecteur):
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

    contenu = (article.get("contenu") or "")
    contenu = " ".join(contenu.split())
    contenu = contenu[:1200]

    # Les erreurs Groq (quota, réseau, aucun modèle) remontent : l'appelant ne marquera pas l'article.
    resultat = selecteur.invoquer(prompt, {
        "titre": article["titre"],
        "source": article["source"],
        "contenu": contenu
    })
    return normaliser_decision(resultat)


def normaliser_decision(texte):
    """Extrait le JSON de la réponse et garantit les 4 clés. Lève ValueError si illisible."""
    m = re.search(r"\{.*\}", texte or "", re.DOTALL)
    if not m:
        raise ValueError(f"pas de JSON dans la réponse : {(texte or '')[:80]!r}")
    brut = json.loads(m.group(0))
    try:
        score = float(brut.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    important = brut.get("important", False)
    if isinstance(important, str):
        important = important.strip().lower() in ("true", "oui", "yes", "1")
    return {
        "important": bool(important),
        "score": max(0, min(10, round(score))),
        "raison": str(brut.get("raison", "")),
        "resume": str(brut.get("resume", "")),
    }

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
    """Fusionne avec le rapport existant (au lieu de l'écraser), dédoublonne par lien."""
    os.makedirs("data", exist_ok=True)
    existant = []
    if os.path.exists(RAPPORT_PATH):
        try:
            with open(RAPPORT_PATH, "r", encoding="utf-8") as f:
                existant = json.load(f)
        except (OSError, ValueError):
            existant = []
    par_lien = {item["article"]["lien"]: item for item in existant if isinstance(item, dict) and "article" in item}
    for item in articles_importants:
        par_lien[item["article"]["lien"]] = item
    fusion = sorted(par_lien.values(), key=lambda it: it["decision"].get("score", 0), reverse=True)[:MAX_RAPPORT]
    with open(RAPPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(fusion, f, ensure_ascii=False, indent=2)
    print(f"  💾 Rapport sauvegardé dans {RAPPORT_PATH} ({len(fusion)} articles)")

def sauvegarder_articles_traites(traites, envoyes):
    """N'ajoute que les articles réellement analysés ; dédoublonne par lien."""
    vus, tous_envoyes = set(), []
    for a in envoyes + traites:
        if a["lien"] not in vus:
            vus.add(a["lien"])
            tous_envoyes.append(a)
    with open(ARTICLES_ENVOYES_PATH, "w", encoding="utf-8") as f:
        json.dump(tous_envoyes, f, ensure_ascii=False, indent=2)

# ============================================================
# FONCTION PRINCIPALE
# ============================================================
def lancer_agent(force=False, limite=None):
    print("=" * 60)
    print("🤖 Démarrage de l'Agent IA de veille Microsoft")
    print("=" * 60)

    nouveaux_articles, envoyes = charger_nouveaux_articles(force=force)

    if limite:
        nouveaux_articles = nouveaux_articles[:limite]

    if not nouveaux_articles:
        if not os.path.exists(RAPPORT_PATH):
            sauvegarder_rapport([])
        print("✅ Aucun article à analyser.")
        return {"analyses": 0, "importants": 0, "echecs": 0}

    selecteur = SelecteurModele("GROQ_MODEL_AGENT", max_tokens=800)
    print(f"🧠 Modèle agent (1er candidat) : {selecteur.modele}")

    articles_importants, traites, echecs = [], [], 0
    print("\n🔍 Analyse des articles en cours...\n")

    for i, article in enumerate(nouveaux_articles):
        print(f"  [{i+1}/{len(nouveaux_articles)}] {article['titre'][:60]}...")
        try:
            decision = evaluer_importance(article, selecteur)
        except RuntimeError as e:          # plus aucun modèle utilisable : inutile de continuer
            print(f"    → ❌ {e}")
            echecs += len(nouveaux_articles) - i
            break
        except Exception as e:
            print(f"    → ⚠️ Échec, l'article sera réanalysé au prochain run : {str(e)[:120]}")
            echecs += 1
            continue

        traites.append(article)
        statut = "⭐ IMPORTANT" if decision["important"] else "⏭️  ignoré"
        print(f"    → {statut} (score: {decision['score']}/10) — {decision['raison'][:60]}")

        if decision["important"] and decision["score"] >= SEUIL_SCORE:
            articles_importants.append({"article": article, "decision": decision})
        if i < len(nouveaux_articles) - 1:
            time.sleep(PAUSE_ENTRE_APPELS)

    afficher_rapport(articles_importants)
    sauvegarder_rapport(articles_importants)
    sauvegarder_articles_traites(traites, envoyes)

    print("=" * 60)
    print(f"✅ Agent terminé : {len(traites)} analysés, {len(articles_importants)} importants, {echecs} échecs")
    print("=" * 60)
    return {"analyses": len(traites), "importants": len(articles_importants), "echecs": echecs}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="réanalyser tous les articles")
    parser.add_argument("--limite", type=int, default=None, help="nombre max d'articles à analyser")
    args = parser.parse_args()
    bilan = lancer_agent(force=args.force, limite=args.limite)
    # Job rouge dans GitHub Actions si rien n'a pu être analysé (au lieu d'un faux succès en 1 s)
    sys.exit(1 if bilan["echecs"] and not bilan["analyses"] else 0)