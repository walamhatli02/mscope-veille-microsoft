import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import json
import os
import sys
# Fix UnicodeEncodeError on Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================
# SOURCES RSS MICROSOFT (URLs corrigées et vérifiées)
# ============================================================
SOURCES = [
    {
        "nom": "Microsoft 365 Blog",
        "url": "https://www.microsoft.com/en-us/microsoft-365/blog/feed/"
    },
    {
        "nom": "Azure Updates",
        "url": "https://www.microsoft.com/releasecommunications/api/v2/azure/rss"
    },
    {
        "nom": "Microsoft Tech Community - Microsoft 365",
        "url": "https://techcommunity.microsoft.com/plugins/custom/microsoft/o365/custom-blog-rss?board=Microsoft365Blog"
    },
    {
        "nom": "Microsoft Tech Community - Teams",
        "url": "https://techcommunity.microsoft.com/plugins/custom/microsoft/o365/custom-blog-rss?board=MicrosoftTeamsBlog"
    },
    {
        "nom": "Microsoft Developer Blog",
        "url": "https://devblogs.microsoft.com/feed/"
    },
    {
        "nom": "Microsoft Corporate Blog",
        "url": "https://blogs.microsoft.com/feed/"
    },
]

# ============================================================
# FONCTION : Collecter les articles depuis un flux RSS
# ============================================================
def collecter_articles(source, limite=10):
    """
    Collecte les articles depuis un flux RSS Microsoft.
    Retourne une liste d'articles avec titre, lien, date et contenu.
    """
    articles = []

    try:
        print(f"\n📡 Collecte depuis : {source['nom']}")
        flux = feedparser.parse(source["url"])

        if not flux.entries:
            print(f"  ⚠️  Aucun article trouvé")
            return articles

        for entree in flux.entries[:limite]:
            # Determine a reliable timestamp for the entry
            ts = None
            if hasattr(entree, 'published_parsed') and entree.published_parsed:
                ts = time.mktime(entree.published_parsed)
            elif hasattr(entree, 'updated_parsed') and entree.updated_parsed:
                ts = time.mktime(entree.updated_parsed)
            else:
                # Fallback: try published/updated strings
                date_str = entree.get('published') or entree.get('updated')
                try:
                    ts = time.mktime(datetime.fromisoformat(date_str.replace('Z', '+00:00')).timetuple()) if date_str else time.time()
                except Exception:
                    ts = time.time()

            date_iso = datetime.fromtimestamp(ts).isoformat()

            article = {
                "source": source['nom'],
                "titre": entree.get("title", "Sans titre"),
                "lien": entree.get("link", ""),
                "date": date_iso,
                "timestamp": int(ts),
                "contenu": ""
            }

            # Essayer d'extraire le contenu complet
            if hasattr(entree, "summary"):
                soup = BeautifulSoup(entree.summary, "html.parser")
                article["contenu"] = soup.get_text(separator=" ", strip=True)
            elif hasattr(entree, "content"):
                soup = BeautifulSoup(entree.content[0].value, "html.parser")
                article["contenu"] = soup.get_text(separator=" ", strip=True)

            # Ignorer les articles sans contenu
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
    """
    Sauvegarde les articles collectés dans un fichier JSON.
    """
    os.makedirs("data", exist_ok=True)

    # Charger les anciens articles s'ils existent
    anciens = []
    if os.path.exists(chemin):
        with open(chemin, "r", encoding="utf-8") as f:
            anciens = json.load(f)

    # Indexer les anciens par lien pour permettre la mise à jour si la nouvelle version est plus récente
    index = {a["lien"]: a for a in anciens}
    nouveaux_list = []
    for a in articles:
        lien = a.get("lien")
        if not lien:
            # skip invalid
            continue
        if lien not in index:
            index[lien] = a
            nouveaux_list.append(a)
        else:
            # comparer timestamps si disponibles
            ancien_ts = index[lien].get("timestamp") or 0
            nouveau_ts = a.get("timestamp") or 0
            if nouveau_ts > ancien_ts:
                index[lien] = a
                nouveaux_list.append(a)

    # Recomposer la liste en conservant l'ordre des anciens puis en ajoutant les nouveaux/updates
    tous = list(index.values())

    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(tous, f, ensure_ascii=False, indent=2)

    nouveaux = nouveaux_list
    print(f"\n💾 {len(nouveaux)} nouveaux articles sauvegardés ({len(tous)} au total)")
    return nouveaux

# ============================================================
# FONCTION PRINCIPALE
# ============================================================
def lancer_collecte():
    """
    Lance la collecte depuis toutes les sources Microsoft.
    """
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


# ============================================================
# LANCEMENT
# ============================================================
if __name__ == "__main__":
    lancer_collecte()