# E-write-analyzer
Linguistic Corpus engineering writings analyzer
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load a pre-trained model for sentence embeddings
# This model will download on first run
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_coherence_score(text: str):
    """
    Calculates a practical coherence score based on sentence-to-sentence similarity.
    """
    # Split text into sentences
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return 0.0 # Not enough sentences to measure coherence

    # Generate embeddings for each sentence
    embeddings = model.encode(sentences)
    
    # Calculate cosine similarity between consecutive sentences
    similarity_scores = []
    for i in range(len(embeddings) - 1):
        vec1 = embeddings[i].reshape(1, -1)
        vec2 = embeddings[i+1].reshape(1, -1)
        similarity = cosine_similarity(vec1, vec2)[0][0]
        similarity_scores.append(similarity)
        
    # The coherence score is the average of these similarities
    coherence_score = np.mean(similarity_scores)
    
    return coherence_score

# --- Example Usage ---
coherent_text = "The system was designed to optimize energy consumption. We used a PID controller for this task. The controller was tuned to minimize steady-state error. The final design achieved a 15% reduction in power usage."

incoherent_text = "The system was designed to optimize energy consumption. The controller was tuned to minimize steady-state error. We used a PID controller for this task. The final design achieved a 15% reduction in power usage."

print(f"Coherent Text Score: {calculate_coherence_score(coherent_text):.4f}")
print(f"Incoherent Text Score: {calculate_coherence_score(incoherent_text):.4f}")
