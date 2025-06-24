# 🎬 Video Scene Splitter with VideoPrism

Este proyecto segmenta automáticamente un video en escenas usando el modelo `VideoPrism` de Google, basado en la similitud semántica entre embeddings.

## 🚀 Cómo usar

### 1. Clona el repositorio y entra al proyecto

```bash
git clone https://github.com/tu-usuario/video-scene-splitter.git
cd video-scene-splitter
```

### 2. Instala dependencias

```bash
pip install -r requirements.txt
```

### 3. Coloca tu video en la carpeta `videos/`

### 4. Ejecuta el script principal

```bash
python main.py --input videos/mi_video.mp4 --output outputs/ --clip-seconds 1.0 --threshold 0.75
```

## 📂 Estructura

- `videos/`: coloca aquí tus videos.
- `outputs/`: clips de escenas generados.
- `src/`: código fuente.

## 🧠 ¿Cómo funciona?

1. El video se divide en clips de duración fija.
2. Cada clip se convierte en un embedding usando `VideoPrism`.
3. Se calcula la similitud entre clips consecutivos.
4. Si la similitud cae por debajo de un umbral, se marca un corte de escena.
5. Se corta el video en esos puntos usando `ffmpeg`.

## 🛠 Créditos

Basado en el modelo open-source [VideoPrism de Google](https://huggingface.co/google/videoprism).
