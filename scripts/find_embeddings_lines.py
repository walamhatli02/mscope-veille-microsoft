import re
files=['indexer.py','chatbot.py','rag.py','test_vectorstore.py']
patterns=[
 'from langchain_huggingface import HuggingFaceEmbeddings',
 'HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")',
 'from langchain_ollama',
 'OllamaEmbeddings',
 'EMBEDDING_MODEL',
 'OLLAMA_BASE_URL'
]
for f in files:
    try:
        with open(f,'r',encoding='utf-8') as fh:
            for i,line in enumerate(fh,1):
                for p in patterns:
                    if p in line:
                        print(f'{f}:{i}: {line.strip()}')
    except FileNotFoundError:
        print(f'{f}: MISSING')
