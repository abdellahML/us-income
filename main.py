from utils.config import df
from utils.baseline import Baseline
from utils.scoring import Scoring
from utils.validation import crosstabing

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


def console_output(baseline_estimator, accuracy_grid, roc_grid):
    print(
        "\n\nWith a basic RandomForestClassifier, accuracy score is :\n{}".format(
            baseline_output
        )
    )
    print("---------------------------------------\n")
    print("With a GridSearchCv with 'accuracy' as scoring")
    print(accuracy_grid)
    print("VALIDATION")
    print("Crosstab matrix")
    print(crosstab_acc)
    print("---------------------------------------\n")
    print("With a GridSearchCv with 'ROC' as scoring")
    print(roc_grid)
    print("VALIDATION")
    print("Crosstab matrix")
    print(crosstab_roc)


console_output(baseline_output, acc_scoring_output, roc_scoring_output)
