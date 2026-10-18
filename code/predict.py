import joblib
from pathlib import Path
import pandas as pd

model = joblib.load(Path(__file__).parent / "model.pkl")

df = pd.read_csv(Path(__file__).parent / "new_wine.csv")

pred = model.predict(df)
print(pred)