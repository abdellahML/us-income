# Validation functions
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def crosstabing(y_true, y_pred):
    return pd.crosstab(
        y_true, y_pred, rownames=["Actual"], colnames=["Prediction"], margins=True
    )


def cls_report(y_true, y_pred):
    return classification_report(y_true, y_pred)


def roc_curve_test(y_test, y_pred_1, y_pred_2):
    # calculate roc curve and plot the different curves
    fpr_1, tpr_1, thresholds_1 = roc_curve(y_test, y_pred_1)
    fpr_2, tpr_2, thresholds_2 = roc_curve(y_test, y_pred_2)

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]
    # calculate no skill roc curve
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
    plt.plot(fpr_1, tpr_1, marker="v", label="RandomForestClassifier_accuracy")
    plt.plot(fpr_2, tpr_2, marker=".", label="RandomForestClassifier_roc")

    # axis labels
    plt.xlabel("FALSE POSITIVE RATE")
    plt.ylabel("TRUE POSITIVE RATE")
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig("graph/ROC_curve.jpg")
    plt.show()
    return


def score_roc_auc(y_test, y_pred_1, y_pred_2):
    # calculate scores
    ns_auc_1 = roc_auc_score(y_test, y_pred_1)
    ns_auc_2 = roc_auc_score(y_test, y_pred_2)
    return [ns_auc_1, ns_auc_2]


def mat_cor(y_test, y_pred):
    mat_cor_1 = matthews_corrcoef(y_test, y_pred)
    return mat_cor_1


def conf_matrix(y_test, y_pred):
    conf = confusion_matrix(y_test, y_pred)
    return conf


def heatmap_conf(conf, title):
    ax = plt.axes()
    sns.heatmap(conf, ax=ax, center=True)
    ax.set_title(title)
    plt.savefig("graph/{}_heatmap.jpg".format(title))
    plt.close()
    return
