import json
import pandas as pd
from pathlib import Path


DATA_PATH = Path("/data/wr153")

txt_list = []
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
                txt_list.append(text.lower().capitalize())
                
df = pd.DataFrame(txt_list)

df_test = df.sample(frac=0.1)
df_train = df.drop(df_test.index, axis=0)

df_train.to_csv(DATA_PATH.joinpath("text", "train.csv"), index=False)
df_test.to_csv(DATA_PATH.joinpath("text", "test.csv"), index=False)