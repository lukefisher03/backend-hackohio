import csv
import pandas as pd
from pandas.core import indexing

df = pd.read_csv("modified_test_dataset.csv")
label = df.label

true_data = df[df['label'] == 'T']
false_data = df[df['label'] == 'F']
unknown_data = df[df['label'] == 'U']

print(true_data.head())
print(false_data.head())
print(unknown_data.head())

print(len(true_data))
print(len(false_data))
print(len(unknown_data))

minlen = min(len(true_data), len(false_data), len(unknown_data))
b_true = true_data[:minlen]
b_false = false_data[:minlen]
b_unk = unknown_data[:minlen]

out = b_true.append(b_false).append(b_unk)
out.to_csv('balanced_dataset.csv', index=False)
