import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
from collections import defaultdict, Counter

# Download stopwords if you haven't already
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Define the size difference percentage threshold at the top
SIZE_DIFFERENCE_THRESHOLD = 500  # You can change this value

class LyricsAnalyzer:
    def __init__(self):
        # Select device: use MPS if available, otherwise fall back to CPU.
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS device for acceleration.")
        else:
            self.device = torch.device("cpu")
            print("MPS not available, using CPU.")
        
        # Initialize the models
        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        self.sentence_transformer_model_mini = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentence_transformer_model_mpnet = SentenceTransformer('all-mpnet-base-v2')
    
    def get_distilbert_embeddings(self, texts):
        # Batch tokenize with padding
        inputs = self.distilbert_tokenizer(texts, return_tensors='pt', truncation=True, 
                                             max_length=128, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.distilbert_model(**inputs)
        # Mean pooling over tokens to get sentence embeddings.
        last_hidden_state = outputs.last_hidden_state
        embeddings = last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()
    
    def get_sentence_transformer_embeddings(self, texts, model='mini'):
        # Choose between the MiniLM and MPNet models.
        if model == 'mini':
            return self.sentence_transformer_model_mini.encode(texts)
        elif model == 'mpnet':
            return self.sentence_transformer_model_mpnet.encode(texts)
        else:
            raise ValueError("Unknown model type. Choose 'mini' or 'mpnet'.")
    
    def fuse_embeddings(self, texts):
        """
        Computes and concatenates the embeddings from DistilBERT,
        SentenceTransformer (MiniLM), and SentenceTransformer (MPNet).
        """
        # Compute embeddings individually.
        emb_distilbert = self.get_distilbert_embeddings(texts)
        emb_mini = self.get_sentence_transformer_embeddings(texts, model='mini')
        emb_mpnet = self.get_sentence_transformer_embeddings(texts, model='mpnet')
        
        # Normalize each set of embeddings.
        emb_distilbert = normalize(np.array(emb_distilbert))
        emb_mini = normalize(np.array(emb_mini))
        emb_mpnet = normalize(np.array(emb_mpnet))
        
        # Concatenate along the feature axis.
        fused_embeddings = np.concatenate([emb_distilbert, emb_mini, emb_mpnet], axis=1)
        return fused_embeddings

    def tune_clustering(self, embeddings, lines, model_type="OPTICS"):
        best_score = -1  # Initialize with a low score
        best_params = None
        best_labels = None

        if model_type == "OPTICS":
            min_samples_values = [2, 3, 4] if len(embeddings) > 5 else [1, 2]
            max_eps_values = [0.5, 0.75, 1.0]

            for min_samples in min_samples_values:
                for max_eps in max_eps_values:
                    try:
                        optics_model = OPTICS(min_samples=min_samples, max_eps=max_eps, cluster_method='xi')
                        labels = optics_model.fit_predict(embeddings)
                        if len(np.unique(labels)) > 1 and len(labels) > 1:
                            score = silhouette_score(embeddings, labels)
                        else:
                            score = -2
                        print(f"OPTICS: min_samples={min_samples}, max_eps={max_eps}, Silhouette={score}")
                        if score > best_score:
                            best_score = score
                            best_params = {'min_samples': min_samples, 'max_eps': max_eps}
                            best_labels = labels
                    except Exception as e:
                        print(f"Error with min_samples={min_samples}, max_eps={max_eps}: {e}")

            print(f"\nBEST OPTICS: {best_params}, Silhouette={best_score}")
            return best_labels, best_params, best_score
        else:
            raise ValueError(f"Unsupported clustering model type: {model_type}")

    def visualize_embeddings(self, embeddings, lines, labels, title="Fused Lyric Embeddings"):
        """
        Uses t-SNE to reduce the fused embeddings to 2D and then visualizes
        them along with their cluster labels.
        """
        emb_array = np.array(embeddings)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(lines) - 1))
        emb_2d = tsne.fit_transform(emb_array)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="viridis", label="Clusters")

        # Build legend
        legend_elements = []
        for label in sorted(np.unique(labels)):
            marker = 'x' if label == -1 else 'o'
            label_text = 'Noise' if label == -1 else f"Cluster {label}"
            legend_elements.append(plt.Line2D([0], [0], marker=marker, linestyle='',
                                               color='w', markerfacecolor=scatter.cmap(scatter.norm(label)),
                                               markersize=8, label=label_text))
        plt.legend(handles=legend_elements, loc="best")
        
        for i, line in enumerate(lines):
            plt.annotate(line, (emb_2d[i, 0], emb_2d[i, 1]), fontsize=9)
            
        plt.title(title)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.savefig("fused_lyric_embeddings.png")
        plt.close()
        print("Visualization saved as fused_lyric_embeddings.png")
    
    def analyze_fused_lyrics(self, lyrics):
        """
        Splits the lyrics, computes the fused embeddings, clusters them,
        and visualizes the results.
        """
        # Split the input into non-empty lines.
        lines = [line.strip() for line in lyrics.split("\n") if line.strip()]
        if not lines:
            print("No lyrics provided for analysis.")
            return
        
        print("Computing fused embeddings by concatenation...")
        fused_embeddings = self.fuse_embeddings(lines)
        
        print("Clustering fused embeddings using OPTICS...")
        labels, params, score = self.tune_clustering(fused_embeddings, lines)
        print("Best clustering parameters:", params)
        print("Silhouette score:", score)
        
        self.visualize_embeddings(fused_embeddings, lines, labels, title="Fused Lyric Embeddings")
        
        # Optionally, you can add further analysis (e.g., ranking clusters, extracting keywords, etc.)
        return labels