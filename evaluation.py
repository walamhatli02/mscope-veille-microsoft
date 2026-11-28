"""Evaluation reproductible de MsScope (recherche, InsoBot, agent).

Usage :
    python evaluation.py recherche                 # sans API : qualite de la recherche
    python evaluation.py recherche --multilingue   # meme test avec un modele multilingue
    python evaluation.py insobot                   # reponses d'InsoBot (API Groq)
    python evaluation.py agent                     # stabilite de l'agent (API Groq)

Ne modifie aucun fichier de data/ ; ecrit uniquement evaluation_resultats.json.
"""
import json
import os
import random
import re
import sys
import time
import unicodedata

from dotenv import load_dotenv

load_dotenv()

ARTICLES_PATH = "data/articles.json"
RAPPORT_PATH = "data/rapport_agent.json"
ENVOYES_PATH = "data/articles_envoyes.json"
INSOMEA_PATH = "insomea_info.txt"
RESULTATS_PATH = "evaluation_resultats.json"
MODELE_BASE = "sentence-transformers/all-MiniLM-L6-v2"
MODELE_MULTI = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REFUS = "je n'ai pas trouve cette information"

# Chaque question vise un article precis du corpus (reconnu par un extrait de son titre).
# "attendu" : groupes de mots dont la reponse doit contenir au moins un mot par groupe.
QUESTIONS = [
    {"titre": "Support for Java 8, 11 and 17 in Azure Functions",
     "fr": "Quand se termine le support de Java 8, 11 et 17 dans Azure Functions ?",
     "en": "When does support for Java 8, 11 and 17 end in Azure Functions?",
     "attendu": [["2027"]]},
    {"titre": "Cost controls for model and agent spending",
     "fr": "Comment limiter les d\u00e9penses des mod\u00e8les et des agents dans Microsoft Foundry ?",
     "en": "How can I control model and agent spending in Microsoft Foundry?",
     "attendu": [["budget", "seuil", "alerte", "bloqu"]]},
    {"titre": "Sweden Central and Italy North",
     "fr": "Dans quelles nouvelles r\u00e9gions Power Pages est-il disponible ?",
     "en": "In which new regions is Power Pages available?",
     "attendu": [["suede", "sweden"], ["italie", "italy"]]},
    {"titre": "Introducing Microsoft Copilot for legal",
     "fr": "Existe-t-il un Copilot pour les juristes ?",
     "en": "Is there a Copilot for legal teams?",
     "attendu": [["jurid", "legal"]]},
    {"titre": "Microsoft Copilot for finance",
     "fr": "Quelles sont les nouveaut\u00e9s de Copilot pour les \u00e9quipes finance ?",
     "en": "What is new in Copilot for finance teams?",
     "attendu": [["financ"]]},
    {"titre": "Claude models in Microsoft Copilot into the EU Data Boundary",
     "fr": "Les mod\u00e8les Claude dans Copilot respectent-ils la fronti\u00e8re des donn\u00e9es de l'Union europ\u00e9enne ?",
     "en": "Are Claude models in Copilot covered by the EU Data Boundary?",
     "attendu": [["claude"], ["europ", "eu data boundary"]]},
    {"titre": "Introducing Microsoft 365 G7",
     "fr": "Qu'est-ce que Microsoft 365 G7 pour les administrations publiques ?",
     "en": "What is Microsoft 365 G7 for government agencies?",
     "attendu": [["g7"]]},
    {"titre": "Magic Quadrant\u2122 for SIEM",
     "fr": "Microsoft est-il leader dans le Magic Quadrant de Gartner pour le SIEM ?",
     "en": "Is Microsoft a Leader in the Gartner Magic Quadrant for SIEM?",
     "attendu": [["leader"]]},
    {"titre": "Microsoft 365 outage lures",
     "fr": "Quelle campagne d'hame\u00e7onnage utilise de fausses pannes de Microsoft 365 ?",
     "en": "Which phishing campaign uses Microsoft 365 outage lures?",
     "attendu": [["panne", "outage", "interruption"]]},
    {"titre": "30 new CRM skills",
     "fr": "Combien de nouvelles comp\u00e9tences CRM de Dynamics 365 sont disponibles pour Copilot ?",
     "en": "How many new Dynamics 365 CRM skills are available for Copilot?",
     "attendu": [["30", "trente"]]},
    {"titre": "Scheduling Operations Agent",
     "fr": "Quel agent aide \u00e0 planifier les interventions dans Dynamics 365 Field Service ?",
     "en": "Which agent helps with scheduling in Dynamics 365 Field Service?",
     "attendu": [["scheduling operations agent"]]},
    {"titre": "Cross-Platform Meeting Connect",
     "fr": "Peut-on rejoindre une r\u00e9union Zoom depuis une salle Teams Rooms ?",
     "en": "Can I join a Zoom meeting from a Teams Room?",
     "attendu": [["zoom"]]},
    {"titre": "Teams Phone Agent",
     "fr": "Qu'est-ce que Teams Phone Agent ?",
     "en": "What is Teams Phone Agent?",
     "attendu": [["appel", "telephon", "call"]]},
    {"titre": "Edge for Business becomes",
     "fr": "Quel est le navigateur d'entreprise le plus utilis\u00e9 selon Microsoft ?",
     "en": "What is the most used enterprise browser according to Microsoft?",
     "attendu": [["edge"]]},
    {"titre": "Maia 300",
     "fr": "Comment s'appelle la derni\u00e8re puce con\u00e7ue par Microsoft ?",
     "en": "What is Microsoft's latest custom chip?",
     "attendu": [["maia"]]},
]

QUESTIONS_INSOMEA = [
    {"fr": "Depuis quand INSOMEA est-elle partenaire de Microsoft ?", "attendu": [["2017"]]},
    {"fr": "INSOMEA propose-t-elle des services de s\u00e9curit\u00e9 manag\u00e9s ?", "attendu": [["mssp", "securite"]]},
    {"fr": "O\u00f9 se trouvent les bureaux principaux d'INSOMEA ?", "attendu": [["tunis"], ["bahrein", "bahrain"]]},
]

# Questions dont la reponse n'est pas dans les donnees : InsoBot doit refuser.
QUESTIONS_HORS_SUJET = [
    "Quel est le prix de Copilot en Tunisie ?",
    "Quelle est la m\u00e9t\u00e9o \u00e0 Tunis demain ?",
    "Qui a gagn\u00e9 la Coupe du monde de football 2022 ?",
    "Quel est le chiffre d'affaires d'INSOMEA en 2025 ?",
    "Combien co\u00fbte une licence Dynamics 365 Business Central en dinars ?",
]


def normaliser(texte):
    """Meme pretraitement que l'application (preprocess_question)."""
    s = str(texte or "").lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = " ".join(s.split())
    s = re.sub(r"[!?]{2,}", "?", s)
    s = re.sub(r"[\.]{2,}", ".", s)
    return s.replace("\u2019", "'")


def contient(reponse, groupes):
    r = normaliser(reponse)
    return all(any(mot in r for mot in groupe) for groupe in groupes)


def charger(chemin):
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def sauver(cle, valeur):
    resultats = charger(RESULTATS_PATH) if os.path.exists(RESULTATS_PATH) else {}
    resultats[cle] = valeur
    with open(RESULTATS_PATH, "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)


def liens_cibles(articles, extrait):
    liens = {a["lien"] for a in articles if extrait in a.get("titre", "")}
    if not liens:
        raise SystemExit("Article introuvable dans le corpus : " + extrait)
    return liens


def construire_index(modele):
    """Reproduit charger_rag() de chatbot.py (meme decoupage, memes identifiants)."""
    import hashlib
    import chromadb
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings = HuggingFaceEmbeddings(model_name=modele)
    client = chromadb.EphemeralClient()
    vs_ms = Chroma(client=client, collection_name="microsoft_articles", embedding_function=embeddings)
    vs_ins = Chroma(client=client, collection_name="insomea_info", embedding_function=embeddings)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs, ids, vus = [], [], set()
    for a in charger(ARTICLES_PATH):
        lien = a.get("lien", "")
        if lien in vus:
            continue
        vus.add(lien)
        for i, chunk in enumerate(splitter.split_text(f"{a['titre']}\n\n{a.get('contenu', '')}")):
            docs.append(Document(page_content=chunk, metadata={"titre": a.get("titre", ""), "lien": lien}))
            ids.append(hashlib.md5(f"{lien}|{i}".encode("utf-8")).hexdigest())
    for debut in range(0, len(docs), 2000):
        vs_ms.add_documents(docs[debut:debut + 2000], ids=ids[debut:debut + 2000])
    with open(INSOMEA_PATH, encoding="utf-8") as f:
        chunks = splitter.split_text(f.read())
    vs_ins.add_documents([Document(page_content=c, metadata={"titre": "INSOMEA", "lien": "https://insomea.com"})
                          for c in chunks], ids=[f"insomea-{i}" for i in range(len(chunks))])
    return vs_ms, vs_ins


def rang(vs, question, liens, k=30):
    """Rang (1 = premier) du premier fragment de l'article attendu, ou None s'il est absent du top k."""
    for position, doc in enumerate(vs.similarity_search(normaliser(question), k=k), start=1):
        if doc.metadata.get("lien") in liens:
            return position
    return None


def resumer(rangs):
    n = len(rangs)
    return {
        "questions": n,
        "top1": sum(1 for r in rangs if r == 1),
        "top4": sum(1 for r in rangs if r and r <= 4),
        "mrr": round(sum(1 / r for r in rangs if r) / n, 3),
    }


def eval_recherche(multilingue=False):
    modele = MODELE_MULTI if multilingue else MODELE_BASE
    print("Modele d'embedding :", modele)
    print("Construction de l'index (1 a 3 minutes)...")
    articles = charger(ARTICLES_PATH)
    vs_ms, _ = construire_index(modele)
    detail, rangs_fr, rangs_en = [], [], []
    for q in QUESTIONS:
        liens = liens_cibles(articles, q["titre"])
        r_fr, r_en = rang(vs_ms, q["fr"], liens), rang(vs_ms, q["en"], liens)
        rangs_fr.append(r_fr)
        rangs_en.append(r_en)
        detail.append({"article": q["titre"], "rang_fr": r_fr, "rang_en": r_en})
        print(f"  FR rang {str(r_fr):>4} | EN rang {str(r_en):>4} | {q['titre'][:60]}")
    res = {"modele": modele, "francais": resumer(rangs_fr), "anglais": resumer(rangs_en), "detail": detail}
    sauver("recherche_multilingue" if multilingue else "recherche", res)
    print("\nRESULTAT (article attendu parmi les 4 extraits transmis au modele) :")
    for langue in ("francais", "anglais"):
        s = res[langue]
        print(f"  {langue:9}: top1 {s['top1']}/{s['questions']}, top4 {s['top4']}/{s['questions']}, MRR {s['mrr']}")


def eval_insobot():
    from langchain_core.prompts import ChatPromptTemplate
    from groq_utils import SelecteurModele

    print("Construction de l'index (1 a 3 minutes)...")
    articles = charger(ARTICLES_PATH)
    vs_ms, vs_ins = construire_index(MODELE_BASE)
    selecteur = SelecteurModele("GROQ_MODEL_CHAT", max_tokens=1024)
    prompt = ChatPromptTemplate.from_template("""Tu es InsoBot, assistant Microsoft pour l'\u00e9quipe commerciale d'INSOMEA en Tunisie.
R\u00e9ponds TOUJOURS en fran\u00e7ais, m\u00eame si la question est mal formul\u00e9e.
Appuie-toi UNIQUEMENT sur les informations INSOMEA et les actualit\u00e9s Microsoft ci-dessous.
Si elles ne permettent pas de r\u00e9pondre, dis clairement \u00ab Je n'ai pas trouv\u00e9 cette information dans la veille MsScope. \u00bb au lieu d'inventer.
R\u00e9ponds de fa\u00e7on claire et concise (2 \u00e0 5 phrases).

Informations INSOMEA: {context_insomea}
Actualit\u00e9s Microsoft: {context_microsoft}
Question: {question}
R\u00e9ponse en fran\u00e7ais:""")

    def formater(docs):
        return "\n\n".join(f"[{d.metadata.get('titre', '')}]\n{d.page_content}" for d in docs)

    def demander(question):
        q = normaliser(question)
        d_ms, d_ins = vs_ms.similarity_search(q, k=4), vs_ins.similarity_search(q, k=3)
        reponse = selecteur.invoquer(prompt, {"context_microsoft": formater(d_ms), "context_insomea": formater(d_ins),
                                              "question": q})
        time.sleep(10)
        return reponse.strip(), {d.metadata.get("lien") for d in d_ms}

    detail = []
    cas = ([("article", q["fr"], q) for q in QUESTIONS] + [("insomea", q["fr"], q) for q in QUESTIONS_INSOMEA]
           + [("hors_sujet", q, None) for q in QUESTIONS_HORS_SUJET])
    for n, (categorie, question, q) in enumerate(cas, start=1):
        try:
            reponse, liens = demander(question)
        except Exception as exc:
            print(f"  [{n}/{len(cas)}] ERREUR : {str(exc)[:120]}")
            detail.append({"categorie": categorie, "question": question, "erreur": str(exc)[:300]})
            continue
        refus = REFUS in normaliser(reponse)
        ligne = {"categorie": categorie, "question": question, "reponse": reponse, "refus": refus}
        if categorie == "article":
            ligne["article_transmis"] = bool(liens & liens_cibles(articles, q["titre"]))
            ligne["elements_attendus"] = contient(reponse, q["attendu"])
        elif categorie == "insomea":
            ligne["elements_attendus"] = contient(reponse, q["attendu"])
        detail.append(ligne)
        print(f"  [{n}/{len(cas)}] {categorie:10} refus={refus!s:5} {question[:55]}")

    ok = [d for d in detail if "erreur" not in d]
    art = [d for d in ok if d["categorie"] == "article"]
    art_transmis = [d for d in art if d["article_transmis"]]
    ins = [d for d in ok if d["categorie"] == "insomea"]
    hs = [d for d in ok if d["categorie"] == "hors_sujet"]
    res = {
        "modele": selecteur.modele,
        "erreurs_api": len(detail) - len(ok),
        "questions_articles": len(art),
        "article_attendu_transmis": len(art_transmis),
        "reponses_justes_quand_article_transmis": sum(1 for d in art_transmis if d["elements_attendus"] and not d["refus"]),
        "refus_quand_article_transmis": sum(1 for d in art_transmis if d["refus"]),
        "refus_quand_article_absent": sum(1 for d in art if not d["article_transmis"] and d["refus"]),
        "questions_insomea": len(ins),
        "reponses_insomea_justes": sum(1 for d in ins if d["elements_attendus"] and not d["refus"]),
        "questions_hors_sujet": len(hs),
        "refus_hors_sujet": sum(1 for d in hs if d["refus"]),
        "detail": detail,
    }
    sauver("insobot", res)
    print("\nRESULTAT :")
    for cle, val in res.items():
        if cle != "detail":
            print(f"  {cle}: {val}")


def eval_agent(taille=10):
    from agent import evaluer_importance
    from groq_utils import SelecteurModele

    articles = charger(ARTICLES_PATH)
    rapport = charger(RAPPORT_PATH)
    envoyes = {a["lien"] for a in charger(ENVOYES_PATH)}
    hasard = random.Random(42)
    importants = hasard.sample(rapport, min(taille, len(rapport)))
    jamais_analyses = [a for a in {a["lien"]: a for a in articles}.values() if a["lien"] not in envoyes]
    nouveaux = hasard.sample(jamais_analyses, min(taille, len(jamais_analyses)))
    echantillon = [(it["article"], it["decision"]) for it in importants] + [(a, None) for a in nouveaux]
    selecteur = SelecteurModele("GROQ_MODEL_AGENT", max_tokens=800)
    print(f"{len(echantillon)} articles x 2 analyses (environ 8 minutes)...")

    detail = []
    for n, (article, ancienne) in enumerate(echantillon, start=1):
        notes = []
        for _ in range(2):
            try:
                d = evaluer_importance(article, selecteur)
                notes.append({"important": d["important"] and d["score"] >= 6, "score": d["score"]})
            except Exception as exc:
                notes.append({"erreur": str(exc)[:200]})
            time.sleep(12)
        ligne = {"titre": article["titre"], "deja_dans_rapport": ancienne is not None,
                 "score_rapport": ancienne["score"] if ancienne else None, "analyse_1": notes[0], "analyse_2": notes[1]}
        detail.append(ligne)
        print(f"  [{n}/{len(echantillon)}] {notes[0].get('score', 'err')!s:>3} / {notes[1].get('score', 'err')!s:>3}"
              f"  (rapport : {ligne['score_rapport']})  {article['titre'][:50]}")

    ok = [d for d in detail if "erreur" not in d["analyse_1"] and "erreur" not in d["analyse_2"]]
    accord = sum(1 for d in ok if d["analyse_1"]["important"] == d["analyse_2"]["important"])
    ecart = [abs(d["analyse_1"]["score"] - d["analyse_2"]["score"]) for d in ok]
    anciens = [d for d in ok if d["deja_dans_rapport"]]
    res = {
        "modele": selecteur.modele,
        "articles": len(detail),
        "analyses_completes": len(ok),
        "accord_entre_deux_analyses": accord,
        "ecart_moyen_score": round(sum(ecart) / len(ecart), 2) if ecart else None,
        "ecart_max_score": max(ecart) if ecart else None,
        "articles_du_rapport_reanalyses": len(anciens),
        "toujours_importants_analyse_1": sum(1 for d in anciens if d["analyse_1"]["important"]),
        "ecart_moyen_avec_rapport": round(sum(abs(d["analyse_1"]["score"] - d["score_rapport"]) for d in anciens)
                                          / len(anciens), 2) if anciens else None,
        "detail": detail,
    }
    sauver("agent", res)
    print("\nRESULTAT :")
    for cle, val in res.items():
        if cle != "detail":
            print(f"  {cle}: {val}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    commande = sys.argv[1] if len(sys.argv) > 1 else ""
    if commande == "recherche":
        eval_recherche(multilingue="--multilingue" in sys.argv)
    elif commande == "insobot":
        eval_insobot()
    elif commande == "agent":
        eval_agent()
    else:
        print(__doc__)
