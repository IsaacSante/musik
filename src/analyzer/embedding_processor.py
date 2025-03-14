from queue import Queue
from threading import Thread
from sentence_transformers import SentenceTransformer
import torch
from src.analyzer.embedding_visualizer import EmbeddingVisualizer

class EmbeddingProcessor:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_queue = Queue()
        
        # Check for MPS (Metal Performance Shaders) availability
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Metal Performance Shaders) acceleration on Apple Silicon")
        else:
            self.device = torch.device("cpu")
            print("MPS not available, falling back to CPU")
            
        # Load model to the appropriate device
        self.model = SentenceTransformer(embedding_model)
        self.model.to(self.device)
        
        self.embeddings = []
        self.worker_thread = Thread(target=self.worker, daemon=True)
        self.worker_thread.start()

    def worker(self):
        while True:
            lyric_data = self.embedding_queue.get()
            if lyric_data is None:
                break
            
            lyric_text = lyric_data.get('lyric')
            subject = lyric_data.get('subject', '')
            concept = lyric_data.get('concept', '')
            emotion = lyric_data.get('emotion', '')

            combined_text = f"{subject} {concept} {emotion}"

            if combined_text.strip():
                try:
                    # Generate embedding using the device-placed model
                    with torch.no_grad():
                        embedding = self.model.encode(combined_text)
                    
                    result = {
                        'lyric': lyric_text,
                        'embedding': embedding,
                        'combined_text': combined_text,
                    }
                    self.embeddings.append(result)
                    print("Generated embedding:", combined_text)
                except Exception as e:
                    print("Error generating embedding:", e)
            self.embedding_queue.task_done()

    def enqueue(self, lyric_data):
        self.embedding_queue.put(lyric_data)

    def shutdown(self, visualize=True, output_file='embeddings.png'):
        # Shutdown worker
        self.embedding_queue.put(None)
        self.worker_thread.join(timeout=2.0)
        print("EmbeddingProcessor worker thread shut down.")

        # Trigger visualization if requested
        if visualize:
            visualizer = EmbeddingVisualizer(self, output_file)
            visualizer.start()
            visualizer.join(timeout=5.0)
            print("Visualization completed.")
    
    def clear_embeddings(self):
        self.embeddings.clear()
        print("Cleared embeddings.")


# at the bottom of embedding_processor.py
global_embedding_processor = EmbeddingProcessor()

def enqueue_lyric(lyric_data):
    global_embedding_processor.enqueue(lyric_data)
