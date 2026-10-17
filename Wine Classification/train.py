from sklearn.datasets import load_wine
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

wine = load_wine(as_frame=True)

X = wine.data
y = wine.target

wine.frame.to_csv(Path(__file__).parent / "wine.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=True,
    stratify=y,
    random_state=42
)

model_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    min_samples_split=2
)

model_tree.fit(X_train, y_train)

y_prediction_test = model_tree.predict(X_test)
y_prediction_train = model_tree.predict(X_train)

print(accuracy_score(y_pred=y_prediction_test, y_true=y_test))
print(accuracy_score(y_pred=y_prediction_train, y_true=y_train))



model_logistic = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("fit_model", LogisticRegression(max_iter=1000, solver="lbfgs", C=1))
            ])

model_logistic.fit(X_train, y_train)

y_prediction_test2 = model_logistic.predict(X_test)
y_prediction_train2 = model_logistic.predict(X_train)

print(accuracy_score(y_pred=y_prediction_test2, y_true=y_test))
print(accuracy_score(y_pred=y_prediction_train2, y_true=y_train))

