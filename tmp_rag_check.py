import sys
sys.path.insert(0, r'C:\Users\loulouu\veille_microsoft')
from chatbot import charger_rag
run_rag = charger_rag()
question = 'Mon client veut travailler à distance ?'
print('START')
for chunk, docs in run_rag(question):
    print('CHUNK_LEN', len(chunk))
    print(chunk[:200])
    print('DOCS', len(docs))
    break
print('DONE')
