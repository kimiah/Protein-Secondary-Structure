# %%
from glob import glob
import pandas as pd
import numpy as np

data = pd.DataFrame(columns=['substr', 'Structure'])
sub_str_len = 7

files = glob('../DSSP/*.dssp')
for f in files:
    df = pd.read_csv(f, delimiter=' ')
    structure = df['Structure']
    amino_acid = df['AA']
    idx = []  # list of indices
    cumsum = [0]
    for i in range(1, len(df)):
        # print(i)
        cumsum.append(cumsum[i - 1] + 1 if structure[i] == structure[i - 1] else 0)

    for i in range(len(df) - sub_str_len):
        if cumsum[i + sub_str_len - 1] == sub_str_len - 1 and \
                (i + sub_str_len == len(cumsum) or cumsum[i + sub_str_len] == 0):
            idx.append(i)
            data = data.append({'substr': ''.join(amino_acid[i: i + sub_str_len]),
                                'Structure': structure[i]}, ignore_index=True)
    print("f: %s" % f)
    print(idx)

data.to_csv('data.csv')

# %% splitting train and test data
df = data.substr.str.split('', expand=True)
columns = df.columns.tolist()
cols_to_use = columns[1:len(columns)-1]
df = df[cols_to_use]
df.columns = ['AA%d' % i for i in range(1, max_len + 1)]
