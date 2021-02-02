# Validation functions
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def crosstabing(y_true, y_pred):
    return pd.crosstab(
        y_true, y_pred, rownames=["Actual"], colnames=["Prediction"], margins=True
    )


def cls_report(y_true, y_pred):
    print(classification_report(y_test, y_pred_accuracy))
    return
