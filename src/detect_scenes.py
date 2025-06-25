from sklearn.metrics.pairwise import cosine_similarity

def detect_scene_changes(embeddings, threshold=0.75):
    scene_changes = [0]  # Siempre comienza desde el primer clip
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        if similarity < threshold:
            scene_changes.append(i + 1)
    return scene_changes
