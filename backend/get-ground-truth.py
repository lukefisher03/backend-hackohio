import pandas as pd
import os
import csv

df = pd.read_csv("balanced_dataset.csv")
label = df.label
content = df.content.tolist()

#optional: We can import the custom facts that I found online and just manually indexed.
df2 = pd.read_csv("researched-ground-truth.csv")
content2 = df2.content.tolist()
print(len(content2))
collected_truth = [item for (i,item) in enumerate(content) if i <= len([l for l in label if l == "T"])-1]
final_truth = collected_truth + content2
print(len(final_truth))

final_df = pd.DataFrame(final_truth, columns=["content","link"])
final_df.to_csv("ground-truth-data", index=False)