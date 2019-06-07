# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import string
# %%
sub_str_len = 7


def transform(input_file_name='data_final', str_len=sub_str_len):
    df = pd.read_csv('data/' + input_file_name + '.csv')
    cols = ['AA%d' % i for i in range(1, str_len + 1)]
    df_cols = df[cols]
    cat_encoder = OneHotEncoder(sparse=False,
                                categories=[list(string.ascii_uppercase) for _ in range(str_len)], handle_unknown='ignore')
    df_1hots = cat_encoder.fit_transform(df_cols)
    mat = df_1hots
    df_1hots = (mat.T[~np.all(mat.T == 0, axis=1)]).T
    ordinal_encoder = OrdinalEncoder()
    labels_encoded = ordinal_encoder.fit_transform(df[['label']])
    # data = np.hstack((df_1hots, labels_encoded))
    data = np.concatenate((df_1hots, labels_encoded), axis=1)
    np.save('data/' + input_file_name + '_transformed.npy', data)
    y = data[:, -1]
    X = data[:, :-1]
    return X, y

# %%
# X, y = transform()
data = np.load('data/var_transformed.npy')
X, y = data[:, :-1], data[:, -1]
n_classes = len(np.unique(y))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
