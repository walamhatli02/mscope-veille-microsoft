import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

VECTORSTORE_PATH = "vectorstore"
LLM_MODEL = get_groq_model()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================================
# ÉTAPE 1 : Charger les deux collections
# ============================================================
def charger_vectorstores():
    print("📂 Chargement des vectorstores...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vs_microsoft = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings,
        collection_name="microsoft_articles"
    )
    vs_insomea = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings,
        collection_name="insomea_info"
    )
    print("  ✅ Collections chargées")
    return vs_microsoft, vs_insomea

# ============================================================
# ÉTAPE 2 : Créer le RAG avec deux retrievers
# ============================================================
def creer_rag(vs_microsoft, vs_insomea):
    print("🔗 Construction du RAG...")

    retriever_microsoft = vs_microsoft.as_retriever(search_kwargs={"k": 4})
    retriever_insomea = vs_insomea.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""Tu es un assistant expert Microsoft pour l'équipe commerciale d'INSOMEA en Tunisie.
Réponds en français en te basant sur le contexte suivant.
Si le contexte contient la réponse, utilise-la directement.
Si tu n'as pas l'information, dis-le clairement sans inventer.

Informations INSOMEA (offres et recommandations):
{context_insomea}

Actualités Microsoft récentes:
{context_microsoft}

Question: {question}

Réponse:""")

    llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.2, max_tokens=220)

    def formater(docs):
        return "\n\n".join([
            f"[{doc.metadata['source']}] {doc.metadata['titre']}\n{doc.page_content}"
            for doc in docs
        ])

    def run_rag(question):
        docs_microsoft = retriever_microsoft.invoke(question)
        docs_insomea = retriever_insomea.invoke(question)

        chain = prompt | llm | StrOutputParser()
        return chain.invoke({
            "context_microsoft": formater(docs_microsoft),
            "context_insomea": formater(docs_insomea),
            "question": question
        }), docs_microsoft, docs_insomea

    print("  ✅ RAG prêt")
    return run_rag

# ============================================================
# ÉTAPE 3 : Poser une question
# ============================================================
def poser_question(run_rag, question):
    print(f"\n❓ Question : {question}")
    print("⏳ Recherche en cours...\n")

    reponse, docs_ms, docs_ins = run_rag(question)

    print("💬 Réponse :\n")
    print(reponse)

    print("\n📚 Sources Microsoft utilisées :")
    for i, doc in enumerate(docs_ms):
        print(f"  [{i+1}] {doc.metadata['titre'][:70]}...")
        print(f"       🔗 {doc.metadata['lien']}")

    return reponse

# ============================================================
# FONCTION PRINCIPALE
# ============================================================
def lancer_rag():
    print("=" * 60)
    print("🚀 Démarrage du RAG Microsoft — INSOMEA")
    print("=" * 60)

    vs_microsoft, vs_insomea = charger_vectorstores()
    run_rag = creer_rag(vs_microsoft, vs_insomea)

    questions = [
        "Mon client veut travailler à distance avec son équipe, quoi lui proposer ?",
        "Quelles sont les dernières mises à jour Azure ?",
    ]

    for question in questions:
        print("\n" + "=" * 60)
        poser_question(run_rag, question)

    print("\n" + "=" * 60)
    print("✅ RAG terminé !")
    print("=" * 60)


if __name__ == "__main__":
    lancer_rag()