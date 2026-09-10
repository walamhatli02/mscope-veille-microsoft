import json
import os
import sys
import shutil
import time
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

ARTICLES_PATH = "data/articles.json"
VECTORSTORE_PATH = "vectorstore"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def charger_articles(chemin=ARTICLES_PATH):
    print("📂 Chargement des articles Microsoft...")
    with open(chemin, "r", encoding="utf-8") as f:
        articles = json.load(f)
    print(f"  ✅ {len(articles)} articles Microsoft chargés")
    return articles

def charger_insomea():
    print("📂 Chargement des infos INSOMEA...")
    if not os.path.exists("insomea_info.txt"):
        print("  ⚠️  insomea_info.txt introuvable")
        return []
    with open("insomea_info.txt", "r", encoding="utf-8") as f:
        contenu = f.read()
    articles = [{
        "source": "INSOMEA",
        "titre": "Informations et offres INSOMEA",
        "lien": "https://insomea.com",
        "date": "2026-01-01",
        "contenu": contenu
    }]
    print("  ✅ Infos INSOMEA chargées")
    return articles

def decouper_en_chunks(articles, chunk_size=500):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    documents = []
    for article in articles:
        texte = f"{article['titre']}\n\n{article['contenu']}"
        chunks = splitter.split_text(texte)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": article["source"],
                    "titre": article["titre"],
                    "lien": article["lien"],
                    "date": article["date"],
                    "chunk": i
                }
            )
            documents.append(doc)
    return documents

def indexer_dans_chroma(docs_microsoft, docs_insomea):
    print("\n🔢 Indexation dans Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(VECTORSTORE_PATH):
        # Fermer proprement la connexion Chroma avant suppression
        try:
            import chromadb
            client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
            client.reset()
            del client
        except Exception:
            pass
        time.sleep(1)
        shutil.rmtree(VECTORSTORE_PATH, ignore_errors=True)
        time.sleep(0.5)
        print("  🗑️  Ancien vectorstore supprimé")

    vs_microsoft = Chroma.from_documents(
        documents=docs_microsoft,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH,
        collection_name="microsoft_articles"
    )
    print(f"  ✅ {len(docs_microsoft)} chunks Microsoft indexés")

    vs_insomea = Chroma.from_documents(
        documents=docs_insomea,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH,
        collection_name="insomea_info"
    )
    print(f"  ✅ {len(docs_insomea)} chunks INSOMEA indexés")

    return vs_microsoft, vs_insomea

def tester_recherche(vs_microsoft, vs_insomea):
    print("\n🔍 Test de recherche...")
    print("\n  [Test Microsoft] Nouveautés Azure :")
    for doc in vs_microsoft.similarity_search("Azure updates", k=3):
        print(f"    ✅ {doc.metadata['titre'][:70]}...")
    print("\n  [Test INSOMEA] Besoin client travail à distance :")
    for doc in vs_insomea.similarity_search("client travail à distance équipe", k=2):
        print(f"    ✅ {doc.page_content[:100]}...")

def lancer_indexation():
    print("=" * 60)
    print("🚀 Démarrage de l'indexation")
    print("=" * 60)

    articles_microsoft = charger_articles()
    articles_insomea = charger_insomea()

    print("\n✂️  Découpage en chunks...")
    docs_microsoft = decouper_en_chunks(articles_microsoft, chunk_size=500)
    docs_insomea = decouper_en_chunks(articles_insomea, chunk_size=300)
    print(f"  ✅ {len(docs_microsoft)} chunks Microsoft")
    print(f"  ✅ {len(docs_insomea)} chunks INSOMEA")

    vs_microsoft, vs_insomea = indexer_dans_chroma(docs_microsoft, docs_insomea)
    tester_recherche(vs_microsoft, vs_insomea)

    print("\n" + "=" * 60)
    print("✅ Indexation terminée ! Deux collections prêtes.")
    print("=" * 60)

    return vs_microsoft, vs_insomea

if __name__ == "__main__":
    lancer_indexation()