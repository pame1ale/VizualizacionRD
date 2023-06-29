import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import dash
from dash import dcc
from dash import html

def read_data(file_path):
    # Leer los datos
    df = pd.read_csv(file_path, header=0)

    # Columnas no numéricas
    non_numeric_columns = df.select_dtypes(exclude=np.number).columns.tolist()

    # Eliminar las columnas no numéricas del conjunto de datos
    df_numeric = df.drop(columns=non_numeric_columns)

    # Valores faltantes
    df_numeric = df_numeric.fillna(df_numeric.mean())
    
    return df_numeric

def pca(data):
    # Normalizar los datos
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(data)

    # Realizar PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(data=pca_data, columns=["PC1", "PC2"])
    
    return pca_df

app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.Link(rel='stylesheet', href='/styles.css'),  # Enlace al archivo CSS
        html.Div(
            className="nav-item",
            style={"width": "15%", "float": "left"},
            children=[
                html.Div(
                    className="nav-link",
                    children=[
                        html.Label(className="text-secondary", children="Dataset"),
                        dcc.Dropdown(
                            id="dataset-dropdown",
                            options=[
                                {"label": "Wine", "value": "wine.data"},
                                {"label": "Airqualityuci", "value": "airqualityuci.csv"},
                                {"label": "Communities", "value": "communities.data"}
                            ],
                            value="wine.data"
                        ),
                    ],
                ),
            ],
        ),

        html.Div(id="pca-graph", style={"width": "85%", "float": "right"}),
    ],
    style={
        "font-family": "Arial, sans-serif",
        "margin": "20px",
    }
)

@app.callback(
    dash.dependencies.Output("pca-graph", "children"),
    [dash.dependencies.Input("dataset-dropdown", "value")]
)
def pca_graph(file_path):
    data = read_data(file_path)
    pca_df = pca(data)
    
    fig = px.scatter(pca_df, x="PC1", y="PC2")
    return dcc.Graph(figure=fig)


if __name__ == "__main__":
    app.run_server(debug=True)
