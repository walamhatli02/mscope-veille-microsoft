from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="vectorstore", embedding_function=embeddings)

resultats = db.similarity_search("Azure OpenShift Confidential Computing", k=6)
for r in resultats:
    print(r.metadata["source"], "—", r.metadata["titre"][:60])