import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from utils.baseline import Baseline


class Scoring(Baseline):
    def __init__(self, df, scoring: str = "accuracy"):
        Baseline.__init__(self, df)
        self.scoring = scoring
        self.df = df
        self.params = {
            "n_estimators": [10, 20, 30, 40, 50, 100],
            "criterion": ["gini", "entropy"],
            "max_depth": [2, 5, 10],
            "min_samples_leaf": [1, 10]
            #'max_features' : np.arange(0.1,1,0.1).tolist()
        }
        self.estimator = RandomForestClassifier(random_state=42)

    def grid_search_scoring_accuracy(self, estimator, params):
        gridsearch = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.params,
            scoring=self.scoring,
            cv=5,  # Use 5 folds
            verbose=1,
            n_jobs=-1
            # Use all but one CPU core
        )
        return gridsearch

    def fit_on_grid_search(self):
        gridsearch = self.grid_search_scoring_accuracy(
            estimator=self.estimator, params=self.params
        )
        accuracy_fit_model = gridsearch.fit(self.X_train, self.y_train)
        # getting best_estimator
        return accuracy_fit_model.best_estimator_

    def run_own(self):
        print("Initializing fitting")
        result = self.fit_on_grid_search()
        # accuracy_best = result.best_estimator_
        pred_on_train = result.predict(self.X_train)
        pred_on_test = result.predict(self.X_test)
        roc_auc_score_ = roc_auc_score(self.y_train, pred_on_train)
        accuracy_score_ = accuracy_score(self.y_train, pred_on_train)

        # output
        score_with_best = result.score(self.X_train, self.y_train)
        output = "\nBest estimator are  :{}%\n".format(result)
        output += "Predictions on train set are : {}%\n".format(pred_on_train)
        output += "Predictions on test set are : {}%\n".format(pred_on_test)
        output += "ROC AUC score on train set is : {}\n".format(roc_auc_score_)
        output += "Accuracy score on tran set is : {}\n".format(accuracy_score_)
        output += (
            "The generalization accuracy of the model is {:.2f}% on train set".format(
                score_with_best * 100
            )
        )

        return output, pred_on_test, self.X_test, self.y_test, result
