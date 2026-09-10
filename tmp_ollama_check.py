# -*- coding: utf-8 -*-
import os
from langchain_groq import ChatGroq

print('before')
llm = ChatGroq(model=os.getenv('GROQ_MODEL', 'groq/compound'), api_key=os.getenv('GROQ_API_KEY'), temperature=0.2, max_tokens=128)
print('calling')
resp = llm.invoke('Bonjour, réponds en une phrase.')
print('response:', resp.content[:200])
