import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def entrenar_clasificador(df, target_col):

    # Separar X e y
    X_data = df.drop(columns=[target_col]).values
    y_data = df[target_col].values

    # Escalar TODO X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)

    # Split secuencial 80/20
    split = int(0.8 * len(df))

    X_train = X_scaled[:split]
    X_test = X_scaled[split:]

    y_train = y_data[:split]

    # Modelo
    model = LogisticRegression()

    model.fit(X_train, y_train)

    # Predicciones
    preds = model.predict(X_test)

    return preds