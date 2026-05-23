import cv2
import csv
import os
from Funciones.mediapipe_mano import crear_detector_mano, crear_imagen_mediapipe, dibujar_puntos_manos
from Funciones.normalizacionCords import obtenerAngulos

LETRAS = ['A', 'B', 'D', 'E', 'F', 'I', 'K', 'L', 'N', 'O', 'P', 'U', 'V', 'W', 'Y']
MUESTRAS_POR_LETRA = 100
ARCHIVO_CSV = 'dataset_señas.csv'

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

marca_tiempo_ms = 0

archivo_nuevo = not os.path.exists(ARCHIVO_CSV)
archivo_csv = open(ARCHIVO_CSV, 'a', newline='')
writer = csv.writer(archivo_csv)
if archivo_nuevo:
    writer.writerow(['angle1', 'angle2', 'angle3', 'angle4', 'angle5', 'angle6', 'letra'])

letra_actual_idx = 0
conteo = 0
capturando = False

print(f"\n=== CAPTURA DE DATASET LSM ===")
print(f"Letras a capturar: {LETRAS}")
print(f"Muestras por letra: {MUESTRAS_POR_LETRA}")
print(f"\nPon la seña de '{LETRAS[letra_actual_idx]}' y presiona ESPACIO para empezar a capturar.")
print("Presiona ESC para salir.\n")

with crear_detector_mano(num_manos=1, confianza_minima=0.75) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        marca_tiempo_ms += 33
        imagen_mp = crear_imagen_mediapipe(frame_rgb)
        results = detector.detect_for_video(imagen_mp, marca_tiempo_ms)

        letra_actual = LETRAS[letra_actual_idx]

        cv2.rectangle(frame, (0, 0), (width, 110), (30, 30, 30), -1)
        cv2.putText(frame, letra_actual, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 150), 4)

        progreso = int((conteo / MUESTRAS_POR_LETRA) * 400)
        cv2.rectangle(frame, (150, 20), (550, 55), (80, 80, 80), -1)
        cv2.rectangle(frame, (150, 20), (150 + progreso, 55), (0, 255, 150), -1)
        cv2.putText(frame, f'{conteo}/{MUESTRAS_POR_LETRA}', (560, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        estado_txt = "CAPTURANDO..." if capturando else "Presiona ESPACIO para capturar"
        color_estado = (0, 255, 0) if capturando else (0, 200, 255)
        cv2.putText(frame, estado_txt, (150, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)

        cv2.rectangle(frame, (0, height - 40), (width, height), (30, 30, 30), -1)
        cv2.putText(frame, f"Letras restantes: {LETRAS[letra_actual_idx:]}  |  ESC = salir",
                    (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        if results.hand_landmarks and capturando:
            try:
                angulosid = obtenerAngulos(results, width, height)[0]
                writer.writerow(angulosid + [letra_actual])
                conteo += 1

                texto_angulos = f"Angulos: {[round(a) for a in angulosid]}"
                cv2.putText(frame, texto_angulos, (10, height - 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                if conteo >= MUESTRAS_POR_LETRA:
                    print(f"✅ '{letra_actual}' completada ({MUESTRAS_POR_LETRA} muestras)")
                    conteo = 0
                    capturando = False
                    letra_actual_idx += 1

                    if letra_actual_idx >= len(LETRAS):
                        print("\n🎉 Dataset completo. Archivo guardado en:", ARCHIVO_CSV)
                        break

                    print(f"\nPon la seña de '{LETRAS[letra_actual_idx]}' y presiona ESPACIO.")
            except Exception as e:
                pass

        if results.hand_landmarks:
            dibujar_puntos_manos(frame, results, width, height)

        cv2.imshow('Captura Dataset LSM', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("\nCaptura interrumpida.")
            break
        elif key == 32:
            if not capturando:
                capturando = True
                print(f"▶ Capturando '{letra_actual}'...")
            else:
                capturando = False
                print("⏸ Pausado.")

archivo_csv.close()
cap.release()
cv2.destroyAllWindows()
print(f"\nDataset guardado en: {ARCHIVO_CSV}")
