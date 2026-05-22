import cv2
from Funciones.condicionales import condicionalesLetras
from Funciones.mediapipe_mano import crear_detector_mano, crear_imagen_mediapipe, dibujar_puntos_manos
from Funciones.normalizacionCords import obtenerAngulos

lectura_actual = 0
marca_tiempo_ms = 0

# cap = cv2.VideoCapture("Letras/Letra_o.mp4")
cap = cv2.VideoCapture(0)

wCam, hCam = 1280, 720
cap.set(3, wCam)
cap.set(4, hCam)

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

            dedos = []
            # pulgar externo angle
            if angulosid[5] > 125:
                dedos.append(1)
            else:
                dedos.append(0)

            # pulgar interno
            if angulosid[4] > 150:
                dedos.append(1)
            else:
                dedos.append(0)

            # 4 dedos
            for id in range(0, 4):
                if angulosid[id] > 90:
                    dedos.append(1)
                else:
                    dedos.append(0)

            TotalDedos = dedos.count(1)
            condicionalesLetras(dedos, frame)

            pinky = obtenerAngulos(results, width, height)[1]
            pinkY = pinky[1] + pinky[0]
            resta = pinkY - lectura_actual
            lectura_actual = pinkY
            print(abs(resta), pinkY, lectura_actual)

            if dedos == [0, 0, 1, 0, 0, 0]:
                if abs(resta) > 30:
                    print("jota en movimento")
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.rectangle(frame, (0, 0), (100, 100), (255, 255, 255), -1)
                    cv2.putText(frame, 'J', (20, 80), font, 3, (0, 0, 0), 2, cv2.LINE_AA)

        # Dibujar los puntos manualmente si el detector encuentra manos
        if results.hand_landmarks:
            dibujar_puntos_manos(frame, results, width, height)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
