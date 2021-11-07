import os
import emot
import csv
import pandas as pd
import emoji
import string

emot_core = emot.core.emot() 
df = pd.read_csv("en_dup.csv")
content = df.content
label = df.label
index = []
edited_content = []
new_item = ""
for i, item in enumerate(content):
    no_http  = [i for i in item.split() if "http" not in i]
    new_item = " ".join(no_http)
    for character in item:
        if character not in string.ascii_lowercase and character not in string.ascii_uppercase and character not in string.punctuation:
            new_item = new_item.replace(character, " ")
    df.content[i] = new_item

for i in range(0,len(content)):
    index.append(i)
df = df.drop(df.columns[[2,3,4]], axis=1)
df.columns = [ "label", "content"]
df["index"] = index
print(df)
df.to_csv("modified_test_dataset.csv", index=False)