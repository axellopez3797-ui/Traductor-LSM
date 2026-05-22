# Configuración de MediaPipe Tasks (compatible con Python 3.13 y mediapipe >= 0.10.30)
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

DIRECTORIO_RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO = DIRECTORIO_RAIZ / "modelos" / "hand_landmarker.task"


def crear_detector_mano(num_manos=2, confianza_minima=0.75):
    """Crea el detector de manos para video en vivo."""
    if not RUTA_MODELO.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo en {RUTA_MODELO}. "
            "Ejecuta: python descargar_modelo_mano.py"
        )
    opciones = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(RUTA_MODELO)),
        running_mode=RunningMode.VIDEO,
        num_hands=num_manos,
        min_hand_detection_confidence=confianza_minima,
    )
    return HandLandmarker.create_from_options(opciones)


def crear_imagen_mediapipe(frame_rgb):
    """Convierte un frame RGB de OpenCV al formato que usa MediaPipe."""
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)


def dibujar_puntos_manos(frame, resultado, ancho, alto):
    """Dibuja los puntos de la mano manualmente con OpenCV."""
    if not resultado.hand_landmarks:
        return
    for mano in resultado.hand_landmarks:
        for punto in mano:
            cx, cy = int(punto.x * ancho), int(punto.y * alto)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
