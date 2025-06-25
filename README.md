# 🎬 Video Scene Splitter Web App

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zkartamx/PrismCuts/blob/main/notebooks/VideoSceneSplitter_Colab_FIXED.ipynb)

Este proyecto es una aplicación web que segmenta automáticamente un video en escenas significativas. Utiliza el modelo `MCG-NJU/videomae-base` para analizar el contenido visual de los videos y detectar cambios de escena basándose en la similitud semántica.

La aplicación cuenta con una interfaz de usuario amigable creada con Gradio, que permite a cualquier persona subir un video, ajustar la sensibilidad de la detección y ver los clips resultantes directamente en el navegador.

## ✨ Características

- **Interfaz Web Intuitiva:** Sube y procesa videos fácilmente a través de una interfaz web simple.
- **Detección de Escenas Ajustable:** Controla la sensibilidad de la detección con un umbral deslizable.
- **Modelo Potente:** Utiliza `VideoMAE` para una comprensión profunda del contenido del video.
- **Procesamiento en Segundo Plano:** La interfaz se mantiene responsiva mientras el video es procesado.
- **Uso Alternativo por Línea de Comandos:** Incluye el script original para usuarios avanzados.

## 🛠️ Instalación

Para poder ejecutar el proyecto, necesitas tener `ffmpeg` instalado en tu sistema, además de las dependencias de Python.

### 1. Instalar FFmpeg

- **En macOS (usando Homebrew):**
  ```bash
  brew install ffmpeg
  ```
- **En Linux (usando apt):**
  ```bash
  sudo apt update && sudo apt install ffmpeg
  ```
- **En Windows:**
  Descarga los binarios desde el [sitio oficial de FFmpeg](https://ffmpeg.org/download.html) y añádelos a tu `PATH`.

### 2. Configurar el Proyecto

```bash
# 1. Clona el repositorio
git clone https://github.com/zkartamx/PrismCuts.git
cd PrismCuts

# 2. Crea y activa un entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate

# 3. Instala las dependencias de Python
pip install -r requirements.txt
```

## 🚀 Cómo Usar la Aplicación Web (Recomendado)

La forma más sencilla de usar el proyecto es a través de la interfaz de Gradio.

1.  **Lanza la aplicación:**
    ```bash
    python app.py
    ```
2.  **Abre tu navegador:** Ve a la URL local que aparece en la consola (normalmente `http://127.0.0.1:7860`).
3.  **Sube tu video**, ajusta el umbral y haz clic en "Dividir en Escenas".

## 💻 Uso por Línea de Comandos (Alternativo)

Si prefieres usar el script original:

```bash
python main.py --input videos/tu_video.mp4 --output outputs/ --threshold 0.95
```
- `--input`: Ruta al video de entrada.
- `--output`: Directorio donde se guardarán los clips.
- `--threshold`: Umbral de similitud (un valor más alto es más estricto).

## 🧠 ¿Cómo Funciona?

1.  El video se divide en clips de 1 segundo.
2.  Cada clip se convierte en un *embedding* (una representación numérica) usando el modelo **`VideoMAE`**.
3.  Se calcula la similitud del coseno entre los embeddings de clips consecutivos.
4.  Si la similitud cae por debajo del umbral definido, se marca un corte de escena.
5.  Finalmente, `ffmpeg` corta el video original en los puntos detectados para generar los clips de salida.

## 🛠 Créditos

- Este proyecto utiliza el modelo open-source [VideoMAE (MCG-NJU/videomae-base)](https://huggingface.co/MCG-NJU/videomae-base).
- La interfaz de usuario fue creada con [Gradio](https://www.gradio.app/).
