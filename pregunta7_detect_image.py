import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output, ClientsideFunction, State
import plotly.graph_objects as go
import os
import dash_bootstrap_components as dbc
import base64
from PIL import Image
import io
import numpy as np
import cv2


# Leemos modelo entrenado
import keras
import os

print(os.getcwd())

# Inicializar la app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

model_emotion = keras.models.load_model(
    "/Users/giancarlo/Desktop/models/emotion_model_v3_preg_1.keras"
)


def load_and_preprocess_image(image_path):
    img = np.array(image_path)
    # img = cv2.resize(img, (48, 48))
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))

    # img = img.astype("float32") / 255.0  # Normalizar a [0, 1]
    return image


# Función para hacer la predicción
def make_prediction(image, model):
    # Preprocesar la imagen
    img = load_and_preprocess_image(image)
    img = np.array([img])
    # Realizar predicción
    prediction = model.predict(img)
    prediction = np.argmax(prediction, axis=1)  # Tomar la clase con mayor probabilidad
    return prediction


# Layout de la app
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H2("Clasificador de Imágenes de Gestos"),
                className="text-center mt-4",
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Seleccionar Modelo:"),
                        dcc.Dropdown(
                            id="model-dropdown",
                            options=[
                                {
                                    "label": "Modelo con 20 epocas",
                                    "value": "model_emotion",
                                },
                            ],
                            value="model_emotion",
                            clearable=False,
                        ),
                    ],
                    width=6,
                )
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Upload(
                            id="upload-image",
                            children=html.Div(
                                ["Arrastra o selecciona una imagen aquí"]
                            ),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                            },
                            accept="image/*",
                        ),
                        html.Div(id="output-image-upload", className="mt-3"),
                    ]
                )
            ]
        ),
    ]
)


# Callbacks para procesar la imagen y realizar la predicción
@app.callback(
    Output("output-image-upload", "children"),
    [Input("upload-image", "contents"), Input("model-dropdown", "value")],
)
def update_output_image(contents, selected_model):
    emotion_ranges = [
        "miedo",
        "tristeza",
        "felicidad",
        "desprecio",
        "sorpresa",
        "disgusto",
        "enojo",
    ]
    if contents is not None:
        # Decodificar la imagen
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))

        model = model_emotion

        # Hacer la predicción
        prediction = make_prediction(image, model)
        print(prediction)
        label = "Gesto detectado: " + emotion_ranges[prediction[0]]

        # Mostrar la imagen y el resultado
        return html.Div(
            [
                html.H5(label),
                html.Img(src=contents, style={"width": "100%", "max-width": "300px"}),
            ]
        )
    return html.Div("Por favor, sube una imagen para clasificar.")


# Ejecutar la app
if __name__ == "__main__":
    app.run_server(debug=True)
