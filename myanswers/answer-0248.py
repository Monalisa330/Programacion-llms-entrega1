import pandas as pd
from sklearn.cluster import KMeans

def segmentar_clientes_kmeans(df, k):

    # Crear copia del DataFrame
    output_df = df.copy()

    # Modelo KMeans
    modelo = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    # Asignar clusters
    output_df["cluster"] = modelo.fit_predict(output_df)

    return output_df