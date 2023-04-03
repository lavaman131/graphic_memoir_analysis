from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)

text = "I REALLY DIDN'T KNOW WHAT TO THINK ABOUT THE VEIL. DEEP DOWN I WAS VERY RELIGIOUS BUT AS A FAMILY WE WERE VERY MODERN AND AVANT-GARDE. AT THE AGE OF SIX I WAS ALREADY SURE I WAS THE LAST PROPHET. THIS WAS A FEW YEARS BEFORE THE REVOLUTION. O' ! I WANTED TO BE A PROPHET... BECAUSE OUR MAID DID NOT EAT WITH US. I WAS BORN WITH RELIGION. BECAUSE MY FATHER HAD A CADILLAC. BEFORE ME THERE HAD BEEN A FEW OTHERS I AM THE LAST PROPHET. A WOMAN? AND, ABOVE ALL, BECAUSE MY GRANDMOTHER'S KNEES ALWAYS ACHED. COME HERE MARJI! HELP ME TO STAND UP DON'T WORRY, SOON YOU WON'T HAVE ANY MORE PAN, YOU'LL SEE."
text = preprocess(text)
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)

ranking = np.argsort(scores)
ranking = ranking[::-1]
print(ranking)
for i in range(scores.shape[0]):
    l = labels[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")
