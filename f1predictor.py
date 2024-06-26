import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import joblib


class F1_predictor:
    def __init__(self) -> None:
        self.top5_model = None
        self.model = None
        self.judge_model = None

    def fit_data_preprocesing(self, data):
        data["scaled_points"] = np.nan_to_num(np.sqrt(data['points']), neginf=0.0)
        data['first_5'] = data['final_position'].apply(lambda x: 1 if x <= 5 else 0)
        return data

    def pred_data_preprocesing(self, data):
        data["scaled_points"] = np.nan_to_num(np.sqrt(data['points']), neginf=0.0)
        return data

    def fit_top5_predictor(self, data:pd.DataFrame):
        groups = data['season'].astype(str) + "_" + data['location']
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(gss.split(data, groups=groups))
        train_df = data.iloc[train_idx]
        test_df = data.iloc[test_idx]
        X_train = train_df[["qualification_position", "last_race_position", "scaled_points", "team_pitstop_time"]]
        y_train = train_df["first_5"]
        X_test = test_df[["qualification_position", "last_race_position", "scaled_points", "team_pitstop_time"]]
        y_test = test_df["first_5"]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        self.top5_model = model

    def fit_position_predictor(self, data:pd.DataFrame):
        data_top5 = data[data["first_5"] == 1]
        groups_top5 = data_top5['season'].astype(str) + "_" + data_top5['location']
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        top5_train_idx, top5_test_idx = next(gss.split(data_top5, groups=groups_top5))
        top5_train_df = data_top5.iloc[top5_train_idx]
        X_top5_train = top5_train_df[["qualification_position", "team_pitstop_time"]]
        y_top5_train = top5_train_df["final_position"]

        logreg = LogisticRegression(multi_class='ovr')
        treeClas = RandomForestClassifier(n_estimators=100)
        knn = KNeighborsClassifier(n_neighbors=5)

        estimators = [('knn', knn), ('randFor', treeClas), ('logreg', logreg)]
        ensemble = VotingClassifier(estimators, voting='hard')
        ensemble.fit(X_top5_train, y_top5_train)
        self.model = ensemble

    def fit_judge_model(self, data:pd.DataFrame):
        data_top5 = data[data["first_5"] == 1]
        groups_top5 = data_top5['season'].astype(str) + "_" + data_top5['location']
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        top5_train_idx, top5_test_idx = next(gss.split(data_top5, groups=groups_top5))
        top5_train_df = data_top5.iloc[top5_train_idx]
        X_top5_train = top5_train_df[["qualification_position", "team_pitstop_time"]]
        y_top5_train = top5_train_df["final_position"]
        linreg = LinearRegression()
        linreg.fit(X_top5_train, y_top5_train)
        self.judge_model = linreg

    def fit(self, data):
        data = self.fit_data_preprocesing(data)
        self.fit_top5_predictor(data)
        data = data.dropna()
        self.fit_position_predictor(data)
        self.fit_judge_model(data)

    def predict(self, data):
        data = self.pred_data_preprocesing(data)
        pred_data = data[["qualification_position", "last_race_position", "scaled_points", "team_pitstop_time"]]
        data['pred_5_proba'] = self.top5_model.predict_proba(pred_data)[:,1]
        data_top5 = data.sort_values(ascending=False ,by=['pred_5_proba']).reset_index(drop=True).head(5)
        data_pred_top5 = data_top5[["qualification_position", "team_pitstop_time"]]
        data_top5['pred_position'] = self.model.predict(data_pred_top5)
        data_top5['judge_position'] = self.judge_model.predict(data_pred_top5)

        data_top5 = data_top5.sort_values(by=['pred_position', 'judge_position']).reset_index(drop=True)
        data_top5['unique_pred_position'] = range(1, len(data_top5) + 1)
        return data_top5

train_data = pd.read_csv('train_dataset.csv')
prod_data = pd.read_csv('prod_dataset.csv')

f1p = F1_predictor()
f1p.fit(train_data)
predictions = f1p.predict(prod_data)[["driverid", "unique_pred_position"]]
# print(predictions)



