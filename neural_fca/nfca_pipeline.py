import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

def load_data():
    df = pd.read_csv('../data_sets/credit/clean_dataset.csv')
    df = df.drop(['ZipCode'], axis=1)
    df.index = df.index.astype(str)
    return df


def binarize_X(df, cat_features, num_features, q_num=3, num_strategy='range', add_negation=False):
    df_bin = pd.DataFrame()
    for column in df.columns:
        if column in set(cat_features):
            df_bin = pd.concat([df_bin, pd.get_dummies(df[column], prefix=column)], axis=1)
        elif column in set(num_features):
            _, bins = pd.qcut(df[column], q=q_num, duplicates='drop', retbins=True)
            if num_strategy == 'range':
                intervals = [(round(a,5),round(b,5)) for a,b in zip(bins[:-1],bins[1:])]
            elif num_strategy == 'semirange':
                intervals = [(-np.inf,round(a,5)) for a in bins[1:-1]]
            else:
                raise NotImplementedError
            col_bin = pd.concat([pd.Series((df[column] <= b) & (df[column] >= a), index=df[column].index, name=column+'_'+str([a,b])) for a, b in intervals], axis=1)
            df_bin = pd.concat([df_bin, col_bin], axis=1)
        else:
            df_bin = pd.concat([df_bin, df[column]], axis = 1)
    if add_negation:
        dummies = [pd.get_dummies(df_bin[col], prefix=col) for col in df_bin.columns]
        df_bin = pd.concat(dummies, axis=1)
    return df_bin.astype(bool)


def split_data(X, y, test_size=0.5, val_size=0.1):
    train_size = 1 - test_size
    val_size = val_size / train_size
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=42, shuffle=True, stratify=y_train_val)

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_concepts(X, y, algo='Sofia', criterion=('f1_score', f1_score)):
    K = FormalContext.from_pandas(X)
    L = ConceptLattice.from_context(K, is_monotone=True, algo=algo)
    name, fn = criterion
    for c in L:
        y_preds = np.zeros(K.n_objects)
        y_preds[list(c.extent_i)] = 1
        c.measures[name] = fn(y, y_preds)
    return K, L