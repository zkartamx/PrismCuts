import argparse
from transformers import AutoProcessor, AutoModel
from src.extract_embeddings import get_video_clips, extract_embeddings
from src.detect_scenes import detect_scene_changes
from src.split_video import split_video
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Ruta al video de entrada')
    parser.add_argument('--output', type=str, required=True, help='Directorio de salida')
    parser.add_argument('--clip-seconds', type=float, default=1.0, help='Duración de cada clip en segundos')
    parser.add_argument('--threshold', type=float, default=0.75, help='Umbral de corte por similitud')

    args = parser.parse_args()

    print("[INFO] Cargando modelo VideoPrism...")
    processor = AutoProcessor.from_pretrained("google/videoprism")
    model = AutoModel.from_pretrained("google/videoprism")

    print("[INFO] Extrayendo clips del video...")
    clips = get_video_clips(args.input, args.clip_seconds)

    print("[INFO] Generando embeddings...")
    embeddings = extract_embeddings(clips, model, processor)

    print("[INFO] Detectando cortes de escena...")
    scene_changes = detect_scene_changes(embeddings, threshold=args.threshold)

    print(f"[INFO] Cortes detectados: {scene_changes}")
    os.makedirs(args.output, exist_ok=True)

    print("[INFO] Cortando el video en escenas...")
    split_video(args.input, scene_changes, args.clip_seconds, args.output)

    print("[FINALIZADO] Video segmentado exitosamente.")

if __name__ == "__main__":
    main()
