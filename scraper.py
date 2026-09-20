import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
import os
import sys

if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================
# TOUTES LES SOURCES RSS MICROSOFT
# ============================================================
SOURCES = [
    # Microsoft 365
    {
        "nom": "Microsoft 365 Blog",
        "url": "https://www.microsoft.com/en-us/microsoft-365/blog/feed/"
    },
    # Azure
    {
        "nom": "Azure Updates",
        "url": "https://www.microsoft.com/releasecommunications/api/v2/azure/rss"
    },
    {
        "nom": "Azure AI Blog",
        "url": "https://techcommunity.microsoft.com/plugins/custom/microsoft/o365/custom-blog-rss?board=AzureAIBlog"
    },
    # Sécurité
    {
        "nom": "Microsoft Security Blog",
        "url": "https://www.microsoft.com/en-us/security/blog/feed/"
    },
    # Power Platform
    {
        "nom": "Microsoft Power Platform Blog",
        "url": "https://cloudblogs.microsoft.com/powerplatform/feed/"
    },
    # Dynamics 365
    {
        "nom": "Microsoft Dynamics 365 Blog",
        "url": "https://cloudblogs.microsoft.com/dynamics365/feed/"
    },
    # Teams
    {
        "nom": "Microsoft Teams Blog",
        "url": "https://techcommunity.microsoft.com/plugins/custom/microsoft/o365/custom-blog-rss?board=MicrosoftTeamsBlog"
    },
    # Developer
    {
        "nom": "Microsoft Developer Blog",
        "url": "https://devblogs.microsoft.com/feed/"
    },
    # Corporate
    {
        "nom": "Microsoft Corporate Blog",
        "url": "https://blogs.microsoft.com/feed/"
    },
    # GitHub
    {
        "nom": "GitHub Blog",
        "url": "https://github.blog/feed/"
    },
    # Windows
    {
        "nom": "Windows Blog",
        "url": "https://blogs.windows.com/feed/"
    },
    # Copilot / AI
    {
        "nom": "Microsoft AI Blog",
        "url": "https://blogs.microsoft.com/ai/feed/"
    },
]

# ============================================================
# FONCTION : Collecter les articles depuis un flux RSS
# ============================================================
def collecter_articles(source, limite=10):
    articles = []
    try:
        print(f"\n📡 Collecte depuis : {source['nom']}")
        flux = feedparser.parse(source["url"])

        if not flux.entries:
            print(f"  ⚠️  Aucun article trouvé")
            return articles

        for entree in flux.entries[:limite]:
            article = {
                "source": source["nom"],
                "titre": entree.get("title", "Sans titre"),
                "lien": entree.get("link", ""),
                "date": entree.get("published", str(datetime.now())),
                "contenu": ""
            }

            if hasattr(entree, "summary"):
                soup = BeautifulSoup(entree.summary, "html.parser")
                article["contenu"] = soup.get_text(separator=" ", strip=True)
            elif hasattr(entree, "content"):
                soup = BeautifulSoup(entree.content[0].value, "html.parser")
                article["contenu"] = soup.get_text(separator=" ", strip=True)

            if article["titre"] and article["contenu"]:
                articles.append(article)
                print(f"  ✅ {article['titre'][:80]}...")

    except Exception as e:
        print(f"  ❌ Erreur pour {source['nom']} : {e}")

    return articles

# ============================================================
# FONCTION : Sauvegarder les articles en JSON
# ============================================================
def sauvegarder_articles(articles, chemin="data/articles.json"):
    os.makedirs("data", exist_ok=True)

    anciens = []
    if os.path.exists(chemin):
        with open(chemin, "r", encoding="utf-8") as f:
            anciens = json.load(f)

    liens_existants = {a["lien"] for a in anciens}
    nouveaux = [a for a in articles if a["lien"] not in liens_existants]
    tous = anciens + nouveaux

    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(tous, f, ensure_ascii=False, indent=2)

    print(f"\n💾 {len(nouveaux)} nouveaux articles sauvegardés ({len(tous)} au total)")
    return nouveaux

# ============================================================
# FONCTION PRINCIPALE
# ============================================================
def lancer_collecte():
    print("=" * 60)
    print("🚀 Démarrage de la collecte des nouveautés Microsoft")
    print("=" * 60)

    tous_les_articles = []

    for source in SOURCES:
        articles = collecter_articles(source, limite=10)
        tous_les_articles.extend(articles)

    nouveaux = sauvegarder_articles(tous_les_articles)

    print("\n" + "=" * 60)
    print(f"✅ Collecte terminée : {len(tous_les_articles)} articles récupérés")
    print("=" * 60)

    return nouveaux


if __name__ == "__main__":
    lancer_collecte()