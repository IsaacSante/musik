import threading
import matplotlib
# Set backend to Agg for non-interactive (no GUI) file saving
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

class EmbeddingVisualizer(threading.Thread):
    def __init__(self, embedding_processor, output_file='embeddings.png'):
        super().__init__(daemon=True)
        self.embedding_processor = embedding_processor
        self.output_file = output_file
        self.output_file_lyrics = output_file.replace('.png', '_lyrics.png')

    def visualize_and_save(self):
        embeddings_data = self.embedding_processor.embeddings
        if not embeddings_data:
            print("No embeddings to visualize yet.")
            return

        embeddings = np.array([data['embedding'] for data in embeddings_data])
        combined_text_labels = [data['combined_text'] for data in embeddings_data]
        lyric_labels = [data['lyric'] for data in embeddings_data]

        # PCA reduction to 2D
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        # First visualization with combined text labels
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

        for i, label in enumerate(combined_text_labels):
            plt.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.7)

        plt.title('Embedding Visualization with Concepts (PCA)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True)
        plt.tight_layout()

        # Save plot directly to file
        plt.savefig(self.output_file)
        plt.close()
        print(f"Saved concept embeddings visualization to {self.output_file}")

        # Second visualization with lyric text labels
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

        for i, label in enumerate(lyric_labels):
            plt.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.7)

        plt.title('Embedding Visualization with Lyrics (PCA)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True)
        plt.tight_layout()

        # Save second plot directly to file
        plt.savefig(self.output_file_lyrics)
        plt.close()
        print(f"Saved lyric embeddings visualization to {self.output_file_lyrics}")

    def run(self):
        print("EmbeddingVisualizer thread started, generating visualization files...")
        self.visualize_and_save()
        print("EmbeddingVisualizer visualization file generation complete.")
