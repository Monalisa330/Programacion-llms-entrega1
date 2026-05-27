import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

def entrenar_modelo_estacional(df, target_col, fecha_col):

    # Copia del DataFrame
    df_proc = df.copy()

    # Convertir columna de fecha
    df_proc[fecha_col] = pd.to_datetime(df_proc[fecha_col])

    # Extraer día de la semana
    day = df_proc[fecha_col].dt.dayofweek

    # Variables cíclicas
    df_proc['dia_sin'] = np.sin(2 * np.pi * day / 7)
    df_proc['dia_cos'] = np.cos(2 * np.pi * day / 7)

    # Separar variables
    X = df_proc.drop(columns=[fecha_col, target_col])
    y = df_proc[target_col]

    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenar modelo Lasso
    model = Lasso(alpha=0.1)

    model.fit(X_scaled, y)

    # Contar coeficientes cercanos a cero
    coef_ceros = np.sum(
        np.isclose(model.coef_, 0, atol=1e-5)
    )

    # Calcular MAE
    mae = mean_absolute_error(
        y,
        model.predict(X_scaled)
    )

    # Retorno final
    return (model, int(coef_ceros), float(mae))