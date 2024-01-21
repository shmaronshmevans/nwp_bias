from sklearn import preprocessing
from sklearn import utils
from sklearn.feature_selection import mutual_info_classif as MIC
import pandas as pd
import numpy as np


def mi_score(the_df):
    the_df = the_df.fillna(-999)
    X = the_df.loc[:, the_df.columns != "target_error"]
    y = the_df["target_error"]
    # convert y values to categorical values
    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(y)
    mi_score = MIC(X, y_transformed)
    df = pd.DataFrame()
    df["feature"] = [n for n in the_df.columns if n != "target_error"]
    df["mi_score"] = mi_score

    df = df[df["mi_score"] > 0.2]
    features = df["feature"].tolist()
    return features
