from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Path to the cleaned dataset
DATA_PATH = Path(__file__).resolve().parent / "wine_quality_cleaned.csv"
RESULTS_PATH = Path(__file__).resolve().parent / "results.txt"


# Load the cleaned dataset and prepare for training.
def load_data():
    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError(f"The dataset at {DATA_PATH} is empty.")

    # Use the target column if present; otherwise assume last column is the label.
    target_col = "quality"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


# A standard logistic regression model
def train_logistic_regression(X_train, y_train, X_test):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=41),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, y_pred


# The tuned and final decision tree model used in the results.
def train_decision_tree(X_train, y_train, X_test):
    model = DecisionTreeClassifier(
        max_depth=7,
        random_state=41,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, y_pred


# The default decision tree model with default hyperparameters.
def train_default_decision_tree(X_train, y_train, X_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, y_pred


# The tuned and final gradient boosting model used in the results.
def train_gradient_boosting(X_train, y_train, X_test):
    model = GradientBoostingClassifier(
        max_depth=2,
        min_samples_leaf=15,
        n_iter_no_change=10,
        validation_fraction=0.1,
        tol=1e-4,
        random_state=41,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, y_pred


# The default gradient boosting model with default hyperparameters.
def train_default_gradient_boosting(X_train, y_train, X_test):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, y_pred


# The tuned and final SVM model used in the results.
def train_svm(X_train, y_train, X_test):
    """Train an SVM model and generate test-set predictions."""
    model = make_pipeline(
        StandardScaler(),
        SVC(C=0.1, gamma=0.001, random_state=41),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, y_pred

# The default SVM model with default hyperparameters.
def train_default_svm(X_train, y_train, X_test):
    model = make_pipeline(
        StandardScaler(),
        SVC(),
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, y_pred


# Train one model and return the formatted results.
def evaluate_model(
    model_name,
    train_function,
    X,
    X_train,
    X_validation,
    X_test,
    y_train,
    y_validation,
    y_test,
    use_stratification,
):
    model, y_pred = train_function(X_train, y_train, X_test)

    y_train_pred = model.predict(X_train)
    y_validation_pred = model.predict(X_validation)
    training_accuracy = accuracy_score(y_train, y_train_pred)
    validation_accuracy = accuracy_score(y_validation, y_validation_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    return f"""Model: {model_name}

    # Formatting the evaluation results for readability.
Stratification: {use_stratification}

Dataset rows: {X.shape[0]}
Features: {X.shape[1]}
Training rows: {X_train.shape[0]}
Validation rows: {X_validation.shape[0]}
Test rows: {X_test.shape[0]}

Training accuracy: {training_accuracy:.4f}
Validation accuracy: {validation_accuracy:.4f}
Test accuracy: {test_accuracy:.4f}
"""


# Write the results to an external text file.
def write_results(results, results_path):
    results_path.write_text(results, encoding="utf-8")


if __name__ == "__main__":
    X, y = load_data()
    use_stratification = True  # Set to False to compare without stratification.

    # Split the dataset into training, validation, and test sets. 80-10-10 split.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=41,
        stratify=y if use_stratification else None,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=41,
        stratify=y_temp if use_stratification else None,
    )

    models = {
        "Logistic Regression": train_logistic_regression,
        "Tuned Decision Tree": train_decision_tree,
        "Default Decision Tree": train_default_decision_tree,
        "Tuned Gradient Boosting": train_gradient_boosting,
        "Default Gradient Boosting": train_default_gradient_boosting,
        "Tuned SVM": train_svm,
        "Default SVM": train_default_svm,
    }

    # Iterate over all models and train them.
    all_results = []
    for model_name, train_function in models.items():
        all_results.append(
            evaluate_model(
                model_name,
                train_function,
                X,
                X_train,
                X_validation,
                X_test,
                y_train,
                y_validation,
                y_test,
                use_stratification,
            )
        )
    write_results("\n\n".join(all_results), RESULTS_PATH)
    print(f"Saved results for {len(models)} models to: {RESULTS_PATH}")

