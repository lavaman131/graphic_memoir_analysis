import json
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
from pathlib import Path
from nltk.corpus import stopwords
from tqdm import tqdm
import torch
import pandas as pd

device = "cuda:0" if torch.cuda.is_available() else "cpu"
task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
DATA_PATH = Path("/data/wr153")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

dataset = []
for f in DATA_PATH.joinpath("text").iterdir():
    if f.suffix == ".json":
        with open(f, "rb") as fp:
            d = json.load(fp)
            ch_title = d.pop(sorted(d, key=int)[0])
            # remove outliers for words
            remove_keys = []
            for key, val in d.items():
                if len(val.split()) <= 5:
                    remove_keys.append(key)
            for key in remove_keys:
                d.pop(key)
            for text in d.values():
                dataset.append(text)

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)
# tokenizer.save_pretrained(MODEL)
model = model.to(device)

counter = {}
for text in tqdm(dataset):
    encoded_input = tokenizer(text, return_tensors='pt', max_length=511, truncation="longest_first").to(device)
    output = model(**encoded_input)
    scores = output[0][0].detach().cpu().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    l = labels[ranking[0]]
    counter[l] = counter.get(l, 0) + 1
    
print(counter)

df = pd.DataFrame(counter, index=[0])
df.to_csv("sentiment_count.csv", index=False)
