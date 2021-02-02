import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class Baseline:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.X = self.df.drop("income", axis=1)
        self.y = self.df.income.values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, random_state=2, test_size=0.3
        )

    def basic_random_forest_classifier(self):
        rfc = RandomForestClassifier(random_state=42)
        rfc.fit(self.X_train, self.y_train)
        baseline_accuracy = rfc.score(self.X_test, self.y_test)
        return baseline_accuracy

    def run(self):
        return self.basic_random_forest_classifier()
