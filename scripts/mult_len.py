# %%

from glob import glob
import pandas as pd

data = pd.DataFrame(columns=['substr', 'Structure'])
max_len = 15
min_len = 2

files = glob('../DSSP/*.dssp')
counter = 0

# %%
for f in files:
    counter += 1
    print(counter)
    df = pd.read_csv(f, delimiter=' ')
    structure = df['Structure']
    amino_acid = df['AA']
    idx = []  # list of indices
    cumsum = [0]
    for i in range(1, len(df)):
        # print(i)
        cumsum.append(cumsum[i - 1] + 1 if structure[i] == structure[i - 1] else 0)

    for i in range(1, len(df)):
        if i == 1 or cumsum[i] == 0:
            str_len = cumsum[i - 1] + 1
            if min_len <= str_len <= max_len:
                idx.append(i - str_len)
                # print("[%d:%d]" % (i - str_len, i))
                # print(''.join(amino_acid[i - str_len: i]) + " " + ''.join(structure[i - str_len: i]))
                data = data.append({
                    # 'substr': '{:<012d}'.format(''.join(amino_acid[i - str_len: i])),
                    'substr': (''.join(amino_acid[i - str_len: i]).zfill(max_len)),
                    'Structure': structure[i - 1]}, ignore_index=True)
    print("f: %s" % f)
    # print()

# %%
df = data.substr.str.split('', expand=True)
columns = df.columns.tolist()
cols_to_use = columns[1:len(columns) - 1]
df = df[cols_to_use]
df.columns = ['AA%d' % i for i in range(max_len)]
