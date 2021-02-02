from utils.config import df
from utils.baseline import Baseline
from utils.scoring import Scoring
from utils.validation import (
    crosstabing,
    cls_report,
    roc_curve_test,
    mat_cor,
    conf_matrix,
    heatmap_conf,
)

# baseline
baseline = Baseline(df)
baseline_output = baseline.run()
# Accuracy
acc_scoring = Scoring(df, scoring="accuracy")
(
    acc_scoring_output,
    prediction_accuracy,
    X_test_acc,
    y_true_acc,
    best_estimator_acc,
) = acc_scoring.run_own()
# Roc
roc_scoring = Scoring(df, scoring="roc")
(
    roc_scoring_output,
    prediction_roc,
    X_test_roc,
    y_true_roc,
    best_estimator_roc,
) = acc_scoring.run_own()
# Validation
crosstab_acc = crosstabing(y_true=y_true_acc, y_pred=prediction_accuracy)
crosstab_roc = crosstabing(y_true=y_true_roc, y_pred=prediction_roc)
cls_report_acc = cls_report(y_true=y_true_acc, y_pred=prediction_accuracy)
cls_report_roc = cls_report(y_true=y_true_roc, y_pred=prediction_roc)
mat_cor_accuracy = mat_cor(y_test=y_true_acc, y_pred=prediction_accuracy)
mat_cor_roc = mat_cor(y_test=y_true_roc, y_pred=prediction_roc)


def console_output(baseline_estimator, accuracy_grid, roc_grid):
    """Function who handle the outpout of the script in a terminal window

    Args:
        baseline_estimator (RandomForestClassifier): RandomClassifier with no  hyper-parameter tuning
        accuracy_grid (RandomForestClassifier): RandomClassifier with a gridSearch with accuracy as scoring
        roc_grid (RandomForestClassifier): RandomClassifier with a gridSearch with roc as scoring

        #All the classifier are initialized with the same random state.
    """
    print(
        "\n\nWith a basic RandomForestClassifier, accuracy score is :\n{}".format(
            baseline_output
        )
    )
    print("---------------------------------------\n")
    print("With a GridSearchCv with 'accuracy' as scoring")
    print(accuracy_grid)
    print("\tVALIDATION")
    print("Crosstab matrix\n")
    print(crosstab_acc)
    print("Classification Report\n")
    print(cls_report_acc)
    print("Matthews Correlation Coefficient")
    print(mat_cor_accuracy)
    print("---------------------------------------\n")
    print("With a GridSearchCv with 'ROC' as scoring")
    print(roc_grid)
    print("\tVALIDATION")
    print("Crosstab matrix")
    print(crosstab_roc)
    print("Classification Report\n")
    print(cls_report_roc)
    print("Matthews Correlation Coefficient")
    print(mat_cor_roc)

    roc_proba = best_estimator_roc.predict_proba(X_test_roc)
    accuracy_proba = best_estimator_acc.predict_proba(X_test_acc)
    roc_curve_test(
        y_test=y_true_acc, y_pred_1=accuracy_proba[:, 1], y_pred_2=roc_proba[:, 1]
    )
    return


console_output(baseline_output, acc_scoring_output, roc_scoring_output)
# heatmap
conf_accuracy = conf_matrix(y_true_acc, prediction_accuracy)
conf_roc = conf_matrix(y_true_roc, prediction_roc)
heatmap_conf(conf=conf_accuracy, title="Accuracy")
heatmap_conf(conf=conf_roc, title="ROC")
