# Traductor de Lengua de Señas (LSM) con Inteligencia Artificial

Aplicación de **visión por computadora** que detecta la posición de las manos en tiempo real y traduce gestos estáticos y dinámicos de la **Lengua de Señas Mexicana (LSM)** a letras en español.

Esta versión es un **fork modificado y optimizado** del proyecto original de dominio público. Fue adaptada para funcionar de forma estable en entornos modernos con **Python 3.13** y **MediaPipe 0.10.35+**, sustituyendo la API obsoleta `mp.solutions` por **MediaPipe Tasks** y corrigiendo errores de renderizado de landmarks (`AttributeError: module 'mediapipe' has no attribute 'solutions'`).

---

## Características

- Detección de manos en vivo con la cámara web
- Cálculo de ángulos entre articulaciones para identificar dedos extendidos
- Reconocimiento de letras estáticas (A, B, D, E, I, O, U, etc.)
- Detección de la letra **J** por movimiento del meñique
- Dibujo manual de landmarks con OpenCV (círculos verdes)
- Compatible con Python 3.13 y versiones recientes de MediaPipe

---

## Requisitos

- Python **3.13** (recomendado)
- Cámara web
- Windows, Linux o macOS

---

## Instalación

Clona o descarga el repositorio, abre una terminal en la carpeta raíz del proyecto y ejecuta estos comandos en orden:

```bash
pip install --only-binary=:all: mediapipe
pip install "numpy<2"
pip install -r requirements.txt
```

Descarga el modelo de manos necesario para MediaPipe Tasks (solo la primera vez):

```bash
python descargar_modelo_mano.py
```

---

## Uso

Ejecuta el script principal desde la raíz del proyecto:

```bash
python "letra en movimiento.py"
```

También puedes usar la versión con detección de más letras:

```bash
python app.py
```

- Pulsa **ESC** para cerrar la ventana de la cámara.

---

## Estructura del proyecto

```
├── app.py                      # Traductor con más letras del abecedario
├── letra en movimiento.py      # Script principal (J en movimiento + I)
├── descargar_modelo_mano.py    # Descarga el modelo .task de MediaPipe
├── requirements.txt            # Dependencias del proyecto
├── modelos/
│   └── hand_landmarker.task    # Modelo de detección de manos
└── Funciones/
    ├── mediapipe_mano.py       # Configuración de MediaPipe Tasks
    ├── normalizacionCords.py   # Cálculo de ángulos y coordenadas
    └── condicionales.py        # Reglas para cada letra
```

---

## Tecnologías

| Herramienta   | Uso                                              |
|---------------|--------------------------------------------------|
| **MediaPipe** | Detección de landmarks de manos (Tasks API)      |
| **OpenCV**    | Captura de video, dibujo y visualización         |
| **NumPy**     | Normalización de coordenadas y cálculo de ángulos |

---

## Licencia

Este proyecto se distribuye bajo **CC0 1.0 Universal (Dominio Público)**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

## Mantenedor

**Axel Lopez** — [@axellopez3797-ui](https://github.com/axellopez3797-ui)

Versión adaptada y mantenida a partir del núcleo de dominio público original (Cesar Ortiz, Jahaziel Hernandez).
