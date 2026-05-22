# Descarga el modelo de manos requerido por MediaPipe Tasks
from pathlib import Path
from urllib.request import urlretrieve

URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
RUTA_DESTINO = Path(__file__).resolve().parent / "modelos" / "hand_landmarker.task"


def main():
    try:
        RUTA_DESTINO.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando modelo en {RUTA_DESTINO}...")
        urlretrieve(URL_MODELO, RUTA_DESTINO)
        print("Descarga completada.")
    except Exception as error:
        print(f"Error al descargar el modelo: {error}")


if __name__ == "__main__":
    main()
