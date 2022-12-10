from datasets import load_dataset
import re

def cleantext(texttype:str = "train"):
    text = kor.sub(' ',str(dataset[texttype][i]['title']))
    text = ' '.join(text.split())
    return text
    

dataset = load_dataset('klue', 'ynat')

idxlist = []
for i in range(len(dataset["train"])):
    idxlist.append(str(i) + '\ttrain\t' + str(dataset['train'][i]['label']))
    
for i in range(len(dataset["validation"])):
    idxlist.append(str(i) + '\ttest\t' + str(dataset['validation'][i]['label']))
    
f = open('data/klue.txt', 'w')
f.write('\n'.join(idxlist))
f.close()

corpuslist = []
kor = re.compile('[^ 一-龥a-zA-Z가-힣+]')
for i in range(len(dataset["train"])):
    corpuslist.append(cleantext("train"))
    
for i in range(len(dataset["validation"])):
    corpuslist.append(cleantext("validation"))
    
f = open('data/corpus/klue.txt', 'w', encoding="UTF-8")
f.write('\n'.join(corpuslist))
f.close()