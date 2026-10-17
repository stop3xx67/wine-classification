from sklearn.datasets import load_wine
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

wine = load_wine(as_frame=True)

X = wine.data
y = wine.target

wine.frame.to_csv(Path(__file__).parent / "wine.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)