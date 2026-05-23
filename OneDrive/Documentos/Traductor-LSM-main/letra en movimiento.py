import cv2
from Funciones.condicionales import condicionalesLetras
from Funciones.mediapipe_mano import crear_detector_mano, crear_imagen_mediapipe, dibujar_puntos_manos
from Funciones.normalizacionCords import obtenerAngulos
import joblib
import numpy as np
from pathlib import Path

lectura_actual = 0
marca_tiempo_ms = 0

# cap = cv2.VideoCapture("Letras/Letra_o.mp4")
cap = cv2.VideoCapture(0)

wCam, hCam = 1280, 720
cap.set(3, wCam)
cap.set(4, hCam)

RUTA_MODELO = Path(__file__).resolve().parent / "modelos" / "modelo_lsm.pkl"
RUTA_ENCODER = Path(__file__).resolve().parent / "modelos" / "label_encoder.pkl"
modelo = joblib.load(RUTA_MODELO)
le = joblib.load(RUTA_ENCODER)

palabra_actual = ""
ultima_letra = ""
frames_misma_letra = 0
FRAMES_PARA_CONFIRMAR = 20

with crear_detector_mano(num_manos=2, confianza_minima=0.75) as detector:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        marca_tiempo_ms += 33
        imagen_mp = crear_imagen_mediapipe(frame_rgb)
        results = detector.detect_for_video(imagen_mp, marca_tiempo_ms)

        if results.hand_landmarks:
            angulosid = obtenerAngulos(results, width, height)[0]
            pinky = obtenerAngulos(results, width, height)[1]

            angulos_array = np.array(angulosid).reshape(1, -1)
            prediccion = modelo.predict(angulos_array)[0]
            letra = le.inverse_transform([prediccion])[0]

            pinkY = pinky[1] + pinky[0]
            lectura_actual = lectura_actual * 0.7 + pinkY * 0.3
            resta = pinkY - lectura_actual

            if letra == 'I' and abs(resta) > 15:
                letra = 'J'

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(frame, (0, 0), (100, 100), (255, 255, 255), -1)
            cv2.putText(frame, letra, (20, 80), font, 3, (0, 0, 0), 2, cv2.LINE_AA)
            print(letra)

            if letra == ultima_letra:
                frames_misma_letra += 1
            else:
                frames_misma_letra = 0
                ultima_letra = letra

            if frames_misma_letra == FRAMES_PARA_CONFIRMAR:
                if letra == 'B':
                    palabra_actual = palabra_actual[:-1]
                else:
                    palabra_actual += letra
                frames_misma_letra = 0

        # Dibujar los puntos manualmente si el detector encuentra manos
        if results.hand_landmarks:
            dibujar_puntos_manos(frame, results, width, height)

        cv2.rectangle(frame, (0, 110), (640, 180), (30, 30, 30), -1)
        cv2.putText(frame, palabra_actual, (10, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 150), 3)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
