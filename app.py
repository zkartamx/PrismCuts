import gradio as gr
import os
import shutil
from src.extract_embeddings import get_video_clips, extract_embeddings
from src.detect_scenes import detect_scene_changes
from src.split_video import split_video
from transformers import VideoMAEFeatureExtractor, VideoMAEModel

# --- 1. Cargar el modelo y el procesador una sola vez ---
print("[INFO] Cargando modelo y procesador...")
PROCESSOR = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
MODEL = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
print("[INFO] Modelo y procesador cargados exitosamente.")

def process_video(video_path, threshold):
    """Función principal que procesa el video y devuelve las escenas.
    
    Args:
        video_path (str): La ruta al video subido.
        threshold (float): El umbral de similitud para detectar cortes.

    Returns:
        list: Una lista de rutas a los videoclips generados.
    """
    print(f"[INFO] Procesando video: {video_path}")
    print(f"[INFO] Umbral de similitud: {threshold}")

    # Crear un directorio de salida temporal para las escenas
    temp_output_dir = "temp_outputs"
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)
    os.makedirs(temp_output_dir)

    # --- Lógica principal del proyecto ---
    clips = get_video_clips(video_path)
    embeddings = extract_embeddings(clips, MODEL, PROCESSOR)
    scene_changes = detect_scene_changes(embeddings, threshold=threshold)
    
    print(f"[INFO] Cortes detectados: {scene_changes}")
    
    split_video(video_path, scene_changes, 1.0, temp_output_dir)

    # Obtener la lista de videos generados
    output_videos = sorted(
        [os.path.join(temp_output_dir, f) for f in os.listdir(temp_output_dir)],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
    )
    
    print(f"[INFO] Videos generados: {output_videos}")
    return output_videos

# --- 2. Crear la interfaz de Gradio ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎬 Video Scene Splitter
    Sube un video y ajusta el umbral de similitud para dividirlo automáticamente en escenas.
    Un umbral más **alto** (cercano a 1.0) es más estricto y detectará menos escenas.
    Un umbral más **bajo** es más sensible y detectará más escenas.
    """)
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Sube tu video")
            threshold_slider = gr.Slider(
                minimum=0.85,
                maximum=1.0,
                step=0.01,
                value=0.95,
                label="Umbral de Similitud",
                info="Más bajo = más sensible a los cambios de escena"
            )
            submit_button = gr.Button("Dividir en Escenas", variant="primary")
        with gr.Column():
            gallery_output = gr.Gallery(
                label="Escenas Resultantes", 
                show_label=True, 
                elem_id="gallery",
                columns=[2], 
                rows=[2], 
                object_fit="contain", 
                height="auto"
            )

    # Chain events to provide immediate feedback to the user
    submit_button.click(
        # Step 1: Disable button, change text, and clear gallery
        fn=lambda: (gr.update(value="Procesando...", interactive=False), None),
        outputs=[submit_button, gallery_output],
        queue=False
    ).then(
        # Step 2: Run the main processing function
        fn=process_video,
        inputs=[video_input, threshold_slider],
        outputs=gallery_output
    ).then(
        # Step 3: Re-enable the button and restore text
        fn=lambda: gr.update(value="Dividir en Escenas", interactive=True),
        outputs=submit_button,
        queue=False
    )

# --- 3. Lanzar la aplicación ---
if __name__ == "__main__":
    demo.launch(debug=True)
