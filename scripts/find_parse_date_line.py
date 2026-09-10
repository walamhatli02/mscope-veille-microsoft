f='chatbot.py'
with open(f,'r',encoding='utf-8') as fh:
    for i,line in enumerate(fh,1):
        if 'def parse_date_article' in line:
            print(f'{f}:{i}: {line.strip()}')
            # print next 20 lines
            for j in range(1,21):
                try:
                    print(f'{i+j}: '+fh.readline().rstrip('\n'))
                except Exception:
                    break
            break
