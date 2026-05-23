# Traductor de Lengua de Señas Mexicana (LSM) con Inteligencia Artificial

Aplicación de visión por computadora que detecta la posición de las manos en tiempo real y traduce gestos del abecedario de la Lengua de Señas Mexicana (LSM) a letras y palabras en español, usando Machine Learning y Procesamiento de Lenguaje Natural.

---

## Características

- Detección de manos en tiempo real con la cámara web
- Cálculo de ángulos entre articulaciones usando la ley de cosenos
- Clasificación de señas mediante un modelo entrenado con scikit-learn (Random Forest)
- Reconocimiento de 15 letras del abecedario LSM: A, B, D, E, F, I, K, L, N, O, P, U, V, W, Y
- Detección dinámica de la letra J por movimiento del meñique
- Acumulación de letras para formar palabras completas en pantalla
- La letra B funciona como borrado (backspace)
- Compatible con Python 3.13 y MediaPipe 0.10.35

---

## Requisitos

- Python 3.13
- Cámara web
- Windows 10/11

---

## Instalación

Abre una terminal en la carpeta raíz del proyecto y ejecuta:

```bash
pip install opencv-python
pip install mediapipe==0.10.35
pip install "numpy<2"
pip install scikit-learn joblib pandas
```

Descarga el modelo de manos de MediaPipe (solo la primera vez):

```bash
python descargar_modelo_mano.py
```

---

## Uso

Ejecuta el script principal:

```bash
python "letra en movimiento.py"
```

- Haz una seña y mantenla quieta ~0.6 segundos para confirmar la letra
- La letra detectada aparece en la esquina superior izquierda
- La palabra formada se muestra en la parte inferior
- Presiona ESC para cerrar

---

## Estructura del proyecto

```
Traductor-LSM-main/
├── letra en movimiento.py      # Script principal del traductor
├── capturar_dataset.py         # Script para capturar datos de entrenamiento
├── dataset_señas.csv           # Dataset con 1,500 muestras de ángulos
├── Traductor_LSM_Demo.ipynb    # Notebook de demostración en Google Colab
├── descargar_modelo_mano.py    # Descarga el modelo .task de MediaPipe
├── requirements.txt            # Dependencias del proyecto
├── modelos/
│   ├── hand_landmarker.task    # Modelo de detección de manos (MediaPipe)
│   ├── modelo_lsm.pkl          # Clasificador entrenado con scikit-learn
│   └── label_encoder.pkl       # Codificador de etiquetas
└── Funciones/
    ├── mediapipe_mano.py       # Configuración de MediaPipe Tasks
    ├── normalizacionCords.py   # Cálculo de ángulos por dedo
    └── condicionales.py        # Lógica de detección por reglas
```

---

## Tecnologías

| Herramienta     | Uso                                                        |
|-----------------|------------------------------------------------------------|
| MediaPipe       | Detección de 21 landmarks de la mano en tiempo real        |
| OpenCV          | Captura de video y visualización                           |
| NumPy           | Cálculo de ángulos con álgebra lineal                      |
| scikit-learn    | Entrenamiento y uso del clasificador de señas              |
| joblib          | Carga del modelo entrenado                                 |
| pandas          | Manejo del dataset de entrenamiento                        |

---

## Autor

**Axel Lopez** — CECyTEN Tepic | Módulo III Inteligencia Artificial | Mayo 2026
