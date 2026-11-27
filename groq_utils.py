"""Sélection du modèle Groq avec repli automatique (404 / modèle décommissionné)."""
import os
import time

from langchain_groq import ChatGroq

# Seul openai/gpt-oss-20b a été vu fonctionner avec la clé actuelle.
# Les autres sont des replis : ils sont sautés automatiquement s'ils renvoient 404/400.
CANDIDATS_PAR_DEFAUT = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]


def candidats(variable_env):
    """Modèle imposé par la variable d'environnement (ex. GROQ_MODEL_AGENT), puis GROQ_MODEL, puis les défauts."""
    liste = []
    for nom in (os.getenv(variable_env), os.getenv("GROQ_MODEL"), *CANDIDATS_PAR_DEFAUT):
        if nom and nom not in liste:
            liste.append(nom)
    return liste


def erreur_modele(exc):
    """True si l'erreur signifie « ce modèle n'est pas utilisable » (on passe au suivant)."""
    texte = str(exc).lower()
    return any(m in texte for m in ("model_not_found", "does not exist", "decommissioned",
                                    "error code: 404", "model_decommissioned"))


def erreur_quota(exc):
    texte = str(exc).lower()
    return "rate_limit" in texte or "error code: 429" in texte


def construire_llm(modele, max_tokens, streaming=False):
    options = {}
    if modele.startswith("openai/gpt-oss"):
        # gpt-oss consomme des tokens de raisonnement sur max_tokens : on limite le raisonnement.
        options["reasoning_effort"] = "low"
    return ChatGroq(model=modele, api_key=os.getenv("GROQ_API_KEY"), temperature=0.2,
                    max_tokens=max_tokens, max_retries=1, streaming=streaming, **options)


class SelecteurModele:
    """Garde le premier modèle qui marche ; passe au suivant sur 404/400 « modèle »."""

    def __init__(self, variable_env, max_tokens):
        self.candidats = candidats(variable_env)
        self.max_tokens = max_tokens
        self.index = 0

    @property
    def modele(self):
        return self.candidats[self.index] if self.index < len(self.candidats) else None

    def llm(self, streaming=False):
        if self.modele is None:
            raise RuntimeError(f"Aucun modèle Groq utilisable parmi {self.candidats}")
        return construire_llm(self.modele, self.max_tokens, streaming=streaming)

    def invoquer(self, prompt, payload, essais_quota=3, pause_quota=10):
        """prompt | llm | str, avec repli de modèle et attente sur 429. Lève l'exception finale."""
        from langchain_core.output_parsers import StrOutputParser
        essai = 0
        while True:
            chaine = prompt | self.llm() | StrOutputParser()
            try:
                return chaine.invoke(payload)
            except Exception as exc:
                if erreur_modele(exc):
                    print(f"  ↪ modèle {self.modele} indisponible, passage au suivant")
                    self.index += 1          # self.llm() lèvera RuntimeError s'il n'en reste plus
                    continue
                if erreur_quota(exc) and essai < essais_quota - 1:
                    essai += 1
                    time.sleep(pause_quota * essai)
                    continue
                raise