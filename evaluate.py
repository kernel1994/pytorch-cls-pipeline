import pandas as pd
from sklearn.metrics import accuracy_score

import cfg


def eval():
    df = pd.read_csv(cfg.prediction)
    acc = accuracy_score(df.values[:, 0], df.values[:, 1])
    print(f'accuracy score: {acc:5f}')
