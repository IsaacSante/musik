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
            # Expanding the range of parameters:
            min_samples_values = list(range(1, 10))
            max_eps_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

            for min_samples in min_samples_values:
                for max_eps in max_eps_values:
                    try:
                        optics_model = OPTICS(min_samples=min_samples, max_eps=max_eps, cluster_method='xi')
                        labels = optics_model.fit_predict(embeddings)
                        # Only compute a silhouette score if more than one cluster is found
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

    def extract_cluster_labels(self, lines, labels):
        """
        For each cluster (ignoring noise), tokenizes all lines,
        removes stopwords and punctuation, and then assigns the cluster
        a label based on the most common token.
        """
        cluster_to_text = defaultdict(list)
        for line, label in zip(lines, labels):
            if label != -1:
                cluster_to_text[label].append(line)
        
        cluster_labels = {}
        for label, texts in cluster_to_text.items():
            combined_text = " ".join(texts).lower()
            tokens = nltk.word_tokenize(combined_text)
            tokens = [token for token in tokens 
                      if token not in stopwords.words('english') and token not in string.punctuation]
            if tokens:
                most_common_token, _ = Counter(tokens).most_common(1)[0]
            else:
                most_common_token = f"Cluster {label}"
            cluster_labels[label] = most_common_token.capitalize()
        return cluster_labels

    def merge_similar_clusters(self, embeddings, labels, distance_threshold=0.5):
        """
        Merges clusters that have centroids closer than a specified distance threshold.
        Noise points (labeled as -1) remain unchanged.
        """
        labels = np.array(labels)
        unique_labels = [label for label in np.unique(labels) if label != -1]
        centroids = {}

        # Compute centroids for each cluster.
        for label in unique_labels:
            centroids[label] = embeddings[labels == label].mean(axis=0)
        
        merged_map = {}
        new_cluster_id = 0

        # Compare each cluster centroid with others.
        for i, label_i in enumerate(unique_labels):
            if label_i in merged_map:
                continue  # Already merged
            merged_map[label_i] = new_cluster_id
            for label_j in unique_labels[i+1:]:
                if label_j in merged_map:
                    continue
                distance = np.linalg.norm(centroids[label_i] - centroids[label_j])
                if distance < distance_threshold:
                    merged_map[label_j] = new_cluster_id
            new_cluster_id += 1

        # Assign new labels while keeping noise (-1) unchanged.
        new_labels = np.array([merged_map[label] if label in merged_map else -1 for label in labels])
        return new_labels

    def visualize_embeddings(self, embeddings, lines, labels, title="Fused Lyric Embeddings"):
        """
        Uses t-SNE to reduce the fused embeddings to 2D and then visualizes
        them along with their cluster labels. The legend is moved to the right side.
        """
        emb_array = np.array(embeddings)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(lines) - 1))
        emb_2d = tsne.fit_transform(emb_array)

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="viridis", label="Clusters")

        cluster_labels = self.extract_cluster_labels(lines, labels)
        legend_elements = []
        for label in sorted(np.unique(labels)):
            marker = 'x' if label == -1 else 'o'
            label_text = 'Noise' if label == -1 else f"Cluster {label}: {cluster_labels.get(label, '')}"
            legend_elements.append(plt.Line2D([0], [0], marker=marker, linestyle='',
                                               color='w', markerfacecolor=scatter.cmap(scatter.norm(label)),
                                               markersize=8, label=label_text))
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        for i, line in enumerate(lines):
            plt.annotate(line, (emb_2d[i, 0], emb_2d[i, 1]), fontsize=9)
            
        plt.title(title)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.tight_layout()
        plt.savefig("fused_lyric_embeddings.png")
        plt.close()
        print("Visualization saved as fused_lyric_embeddings.png")
    
    def analyze_fused_lyrics(self, lyrics, merge_clusters=True, merge_distance_threshold=0.5):
        """
        Splits the lyrics, computes the fused embeddings, clusters them,
        optionally merges similar clusters, and visualizes the results.
        """
        lines = [line.strip() for line in lyrics.split("\n") if line.strip()]
        if not lines:
            print("No lyrics provided for analysis.")
            return
        
        print("Computing fused embeddings by concatenation...")
        fused_embeddings = self.fuse_embeddings(lines)
        
        print("Clustering fused embeddings using OPTICS...")
        labels, params, score = self.tune_clustering(fused_embeddings, lines)
        if labels is None:
            print("Clustering did not produce valid labels; assigning default cluster (-1) to all lines.")
            labels = [-1] * len(lines)
        print("Best clustering parameters:", params)
        print("Silhouette score:", score)
        
        if merge_clusters:
            print("Merging similar clusters...")
            labels = self.merge_similar_clusters(fused_embeddings, labels, distance_threshold=merge_distance_threshold)
        
        self.visualize_embeddings(fused_embeddings, lines, labels, title="Fused Lyric Embeddings")
        return labels