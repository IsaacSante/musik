from queue import Queue, Empty
from threading import Thread
from sentence_transformers import SentenceTransformer
import torch
from src.analyzer.embedding_visualizer import EmbeddingVisualizer
import time

class EmbeddingProcessor:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", batch_size: int = 10):
        self.embedding_queue = Queue()
        self.batch_size = batch_size
        
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
            # Collect up to batch_size items from the queue
            batch = []
            
            # Get the first item (blocking)
            item = self.embedding_queue.get()
            if item is None:  # Shutdown signal
                self.embedding_queue.task_done()
                break
                
            batch.append(item)
            
            # Try to get more items up to batch_size (non-blocking)
            try:
                for _ in range(self.batch_size - 1):
                    item = self.embedding_queue.get_nowait()
                    if item is None:  # Shutdown signal
                        self.embedding_queue.task_done()
                        # Process remaining batch items first
                        break
                    batch.append(item)
            except Empty:  # Use the imported Empty exception
                # Queue is empty, process what we have
                pass
                
            # Process the batch
            self.process_batch(batch)
            
            # Mark all items in the batch as done
            for _ in range(len(batch)):
                self.embedding_queue.task_done()

    def process_batch(self, batch):
        if not batch:
            return
            
        # Extract text for embedding
        texts = []
        for lyric_data in batch:
            subject = lyric_data.get('subject', '')
            concept = lyric_data.get('concept', '')
            emotion = lyric_data.get('emotion', '')
            combined_text = f"{subject} {concept} {emotion}"
            texts.append(combined_text.strip())
        
        # Filter out empty texts
        valid_indices = [i for i, t in enumerate(texts) if t]
        valid_texts = [texts[i] for i in valid_indices]
        valid_lyric_data = [batch[i] for i in valid_indices]
        
        if not valid_texts:
            return
            
        try:
            start_time = time.time()
            # Generate embeddings for all texts in one call
            with torch.no_grad():
                embeddings = self.model.encode(valid_texts)
            
            # Add the results to self.embeddings
            for i, embedding in enumerate(embeddings):
                result = {
                    'lyric': valid_lyric_data[i]['lyric'],
                    'embedding': embedding,
                    'combined_text': valid_texts[i],
                }
                self.embeddings.append(result)
                
            end_time = time.time()
            print(f"Generated {len(valid_texts)} embeddings in batch ({end_time - start_time:.2f}s)")
        except Exception as e:
            print(f"Error generating embeddings batch: {e}")

    def enqueue(self, lyric_data):
        self.embedding_queue.put(lyric_data)

    def shutdown(self, visualize=True, output_file='embeddings.png', cluster=True, radius=0.5):
        self.embedding_queue.put(None)
        self.worker_thread.join(timeout=2.0)
        print("EmbeddingProcessor worker thread shut down.")

        # Perform clustering if requested
        if cluster and self.embeddings:
            from src.analyzer.embedding_cluster_adapter import EmbeddingClusterAdapter
            cluster_adapter = EmbeddingClusterAdapter(
                self, 
                radius=radius, 
                output_file=output_file.replace('.png', '_clusters.png')
            )
            cluster_adapter.start()
            cluster_adapter.join(timeout=10.0)
            print("Clustering completed.")

        # Trigger visualization if requested
        if visualize:
            from src.analyzer.embedding_visualizer import EmbeddingVisualizer
            visualizer = EmbeddingVisualizer(self, output_file)
            visualizer.start()
            visualizer.join(timeout=5.0)
            print("Visualization completed.")

    
    def clear_embeddings(self):
        self.embeddings.clear()
        print("Cleared embeddings.")


# at the bottom of embedding_processor.py
global_embedding_processor = EmbeddingProcessor(batch_size=5)

def enqueue_lyric(lyric_data):
    global_embedding_processor.enqueue(lyric_data)
