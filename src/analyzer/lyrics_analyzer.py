import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import hdbscan
import nltk
from nltk.corpus import stopwords
import string
from collections import defaultdict, Counter

# Download stopwords if you haven't already
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

class LyricsAnalyzer:
    def __init__(self):
        # Select device: use MPS if available, otherwise fall back to CPU.
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS device for acceleration.")
        else:
            self.device = torch.device("cpu")
            print("MPS not available, using CPU.")
        
        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        self.sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def get_distilbert_embeddings(self, texts):
        # Batch tokenize with padding so that all texts are the same length.
        inputs = self.distilbert_tokenizer(texts, return_tensors='pt', truncation=True, 
                                             max_length=128, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.distilbert_model(**inputs)
        # Mean pooling over tokens to get sentence embeddings.
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        embeddings = last_hidden_state.mean(dim=1)       # (batch_size, hidden_dim)
        return embeddings.cpu().numpy()
    
    def get_sentence_transformer_embeddings(self, texts):
        # SentenceTransformer can process a batch directly.
        return self.sentence_transformer_model.encode(texts)
    
    def tune_clustering(self, embeddings, lines, model_type="DBSCAN"):
        best_score = -1  # Initialize with a low score
        best_params = None
        best_labels = None

        if model_type == "DBSCAN":
            # Baseline eps guess: mean distance to the nearest neighbor.
            nearest_neighbors = NearestNeighbors(n_neighbors=min(5, len(embeddings) - 1))
            nearest_neighbors.fit(embeddings)
            distances, _ = nearest_neighbors.kneighbors(embeddings)
            avg_nearest_neighbor_dist = np.mean(distances[:, 1])
            eps_values = [avg_nearest_neighbor_dist * scale for scale in [0.5, 0.75, 1.0, 1.25]]
            min_samples_values = [2, 3, 4] if len(embeddings) > 5 else [1, 2]

            for eps in eps_values:
                for min_samples in min_samples_values:
                    try:
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        labels = dbscan.fit_predict(embeddings)
                        if len(np.unique(labels)) > 1 and len(labels) > 1:
                            score = silhouette_score(embeddings, labels)
                        else:
                            score = -2
                        print(f"DBSCAN: eps={eps}, min_samples={min_samples}, Silhouette={score}")
                        if score > best_score:
                            best_score = score
                            best_params = {'eps': eps, 'min_samples': min_samples}
                            best_labels = labels
                    except Exception as e:
                        print(f"Error with eps={eps}, min_samples={min_samples}: {e}")

            print(f"\nBEST DBSCAN: {best_params}, Silhouette={best_score}")
            return best_labels, best_params, best_score

        elif model_type == "HDBSCAN":
            min_cluster_size_values = [2, 3, 5]
            cluster_selection_epsilon_values = [0.0, 0.1, 0.2]

            for min_cluster_size in min_cluster_size_values:
                for cluster_selection_epsilon in cluster_selection_epsilon_values:
                    try:
                        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                                        cluster_selection_epsilon=cluster_selection_epsilon)
                        labels = hdbscan_model.fit_predict(embeddings)
                        if len(np.unique(labels)) > 1 and len(labels) > 1:
                            score = silhouette_score(embeddings, labels)
                        else:
                            score = -2
                        print(f"HDBSCAN: min_cluster_size={min_cluster_size}, "
                              f"cluster_selection_epsilon={cluster_selection_epsilon}, Silhouette={score}")
                        if score > best_score:
                            best_score = score
                            best_params = {'min_cluster_size': min_cluster_size, 
                                           'cluster_selection_epsilon': cluster_selection_epsilon}
                            best_labels = labels
                    except Exception as e:
                        print(f"Error with min_cluster_size={min_cluster_size}, "
                              f"cluster_selection_epsilon={cluster_selection_epsilon}: {e}")

            print(f"\nBEST HDBSCAN: {best_params}, Silhouette={best_score}")
            return best_labels, best_params, best_score

        elif model_type == "OPTICS":
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

    def rank_clusterings(self, embeddings, lines, labels):
        """
        Ranks the clustering based on metrics derived from a 2D projection of the embeddings.
        Returns a dictionary with dispersion, coherence, and spread.
        """
        emb_array = np.array(embeddings)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(lines) - 1))
        emb_2d = tsne.fit_transform(emb_array)

        dispersion = np.mean(np.linalg.norm(emb_2d - np.mean(emb_2d, axis=0), axis=1))
        # Coherence: average intra-cluster distance (skip noise cluster)
        cluster_centers = defaultdict(list)
        for i, label in enumerate(labels):
            cluster_centers[label].append(emb_2d[i])
        coherence = 0
        count = 0
        for label, points in cluster_centers.items():
            if label == -1 or len(points) < 2:
                continue
            center = np.mean(points, axis=0)
            coherence += np.mean(np.linalg.norm(points - center, axis=1))
            count += 1
        coherence = coherence / count if count else float('inf')
        # Spread: penalizes the largest cluster's spread on dimension 2.
        cluster_sizes = {label: np.sum(labels == label) for label in np.unique(labels)}
        largest_cluster_id = max(cluster_sizes, key=cluster_sizes.get)
        if largest_cluster_id == -1 or cluster_sizes[largest_cluster_id] <= 1:
            spread = 0
        else:
            largest_cluster_points = emb_2d[labels == largest_cluster_id]
            center = np.mean(largest_cluster_points, axis=0)
            distances = np.abs(largest_cluster_points[:, 1] - center[1])
            spread = -np.max(distances)
        return {'dispersion': dispersion, 'coherence': coherence, 'spread': spread}

    def cluster_lyrics(self, embeddings, lines, model_name, labels):
        """Prints cluster results based on pre-computed labels."""
        print(f"\nCluster Results for {model_name}:")
        for i, line in enumerate(lines):
            print(f"Cluster {labels[i]}: {line}")
        return labels

    def visualize_embeddings(self, embeddings, lines, model_name, labels):
        """Visualizes embeddings using t-SNE with provided labels."""
        emb_array = np.array(embeddings)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(lines) - 1))
        emb_2d = tsne.fit_transform(emb_array)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="viridis")

        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      label=f"Cluster {i}",
                                      markerfacecolor=scatter.cmap(scatter.norm(i)),
                                      markersize=8)
                           for i in np.unique(labels) if i != -1]

        if -1 in np.unique(labels):
            legend_elements.append(plt.Line2D([0], [0], marker='x', color='w',
                                              label='Noise', markerfacecolor='gray', markersize=8))
        plt.legend(handles=legend_elements, loc="best")

        for i, line in enumerate(lines):
            plt.annotate(line, (emb_2d[i, 0], emb_2d[i, 1]), fontsize=9)

        plt.title(f"t-SNE visualization of lyric embeddings - {model_name}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")

        plt.savefig(f"lyric_embeddings_{model_name}.png")
        plt.close()

    def generate_cluster_names(self, labels, lines):
        """Generates names for each cluster based on frequent words."""
        cluster_names = {}
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                cluster_names[cluster_id] = "Noise"
                continue
            cluster_lines = [lines[i] for i, label in enumerate(labels) if label == cluster_id]
            all_words = ' '.join(cluster_lines).lower()
            words = [word for word in all_words.split() if word not in string.punctuation]
            word_counts = Counter(words)
            most_common_words = [word for word, count in word_counts.most_common(3)]
            cluster_names[cluster_id] = ", ".join(most_common_words)
        return cluster_names

    def extract_keywords(self, labels, lines):
        """Extracts keywords for each cluster using TF-IDF."""
        cluster_keywords = {}
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                cluster_keywords[cluster_id] = ["noise"]
                continue
            cluster_lines = [lines[i] for i, label in enumerate(labels) if label == cluster_id]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(cluster_lines)
            term_sums = tfidf_matrix.sum(axis=0)
            term_scores = [(word, term_sums[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            term_scores = sorted(term_scores, key=lambda x: x[1], reverse=True)
            top_keywords = [word for word, score in term_scores[:5]]
            cluster_keywords[cluster_id] = top_keywords
            print(f"Cluster {cluster_id} Keywords: {top_keywords}")
        return cluster_keywords

    def analyze_lyrics(self, lyrics):
        lines = [line.strip() for line in lyrics.split("\n") if line.strip()]
        if not lines:
            print("No lyrics to analyze.")
            return

        # Compute DistilBERT embeddings in batch.
        print("Computing DistilBERT embeddings in batch...")
        distilbert_embeddings = self.get_distilbert_embeddings(lines)
        distilbert_embeddings = normalize(np.array(distilbert_embeddings))

        # Compute SentenceTransformer embeddings in batch.
        print("Computing SentenceTransformer embeddings in batch...")
        sentence_transformer_embeddings = self.get_sentence_transformer_embeddings(lines)
        sentence_transformer_embeddings = normalize(np.array(sentence_transformer_embeddings))

        # ---------------------------
        # DBSCAN Clustering
        # ---------------------------
        print("DistilBERT Clustering with DBSCAN:")
        distilbert_labels_dbscan, distilbert_params_dbscan, distilbert_score_dbscan = self.tune_clustering(distilbert_embeddings, lines, "DBSCAN")
        
        print("\nSentenceTransformer Clustering with DBSCAN:")
        sentence_transformer_labels_dbscan, sentence_transformer_params_dbscan, sentence_transformer_score_dbscan = self.tune_clustering(sentence_transformer_embeddings, lines, "DBSCAN")
        
        # Print and visualize DBSCAN results.
        self.cluster_lyrics(distilbert_embeddings, lines, "DistilBERT DBSCAN", distilbert_labels_dbscan)
        self.visualize_embeddings(distilbert_embeddings, lines, "DistilBERT DBSCAN", distilbert_labels_dbscan)
        self.cluster_lyrics(sentence_transformer_embeddings, lines, "SentenceTransformer DBSCAN", sentence_transformer_labels_dbscan)
        self.visualize_embeddings(sentence_transformer_embeddings, lines, "SentenceTransformer DBSCAN", sentence_transformer_labels_dbscan)

        # ---------------------------
        # HDBSCAN Clustering
        # ---------------------------
        print("\nDistilBERT Clustering with HDBSCAN:")
        distilbert_labels_hdbscan, distilbert_params_hdbscan, distilbert_score_hdbscan = self.tune_clustering(distilbert_embeddings, lines, "HDBSCAN")
        
        print("\nSentenceTransformer Clustering with HDBSCAN:")
        sentence_transformer_labels_hdbscan, sentence_transformer_params_hdbscan, sentence_transformer_score_hdbscan = self.tune_clustering(sentence_transformer_embeddings, lines, "HDBSCAN")
        
        # Print and visualize HDBSCAN results.
        self.cluster_lyrics(distilbert_embeddings, lines, "DistilBERT HDBSCAN", distilbert_labels_hdbscan)
        self.visualize_embeddings(distilbert_embeddings, lines, "DistilBERT HDBSCAN", distilbert_labels_hdbscan)
        self.cluster_lyrics(sentence_transformer_embeddings, lines, "SentenceTransformer HDBSCAN", sentence_transformer_labels_hdbscan)
        self.visualize_embeddings(sentence_transformer_embeddings, lines, "SentenceTransformer HDBSCAN", sentence_transformer_labels_hdbscan)

        # ---------------------------
        # OPTICS Clustering
        # ---------------------------
        print("\nDistilBERT Clustering with OPTICS:")
        distilbert_labels_optics, distilbert_params_optics, distilbert_score_optics = self.tune_clustering(distilbert_embeddings, lines, "OPTICS")
        
        print("\nSentenceTransformer Clustering with OPTICS:")
        sentence_transformer_labels_optics, sentence_transformer_params_optics, sentence_transformer_score_optics = self.tune_clustering(sentence_transformer_embeddings, lines, "OPTICS")
        
        # Print and visualize OPTICS results.
        self.cluster_lyrics(distilbert_embeddings, lines, "DistilBERT OPTICS", distilbert_labels_optics)
        self.visualize_embeddings(distilbert_embeddings, lines, "DistilBERT OPTICS", distilbert_labels_optics)
        self.cluster_lyrics(sentence_transformer_embeddings, lines, "SentenceTransformer OPTICS", sentence_transformer_labels_optics)
        self.visualize_embeddings(sentence_transformer_embeddings, lines, "SentenceTransformer OPTICS", sentence_transformer_labels_optics)

        # ---------------------------
        # Ranking Clustering Results
        # ---------------------------
        # Combine silhouette scores from all methods.
        all_rankings = {
            "DistilBERT_DBSCAN": distilbert_score_dbscan,
            "SentenceTransformer_DBSCAN": sentence_transformer_score_dbscan,
            "DistilBERT_HDBSCAN": distilbert_score_hdbscan,
            "SentenceTransformer_HDBSCAN": sentence_transformer_score_hdbscan,
            "DistilBERT_OPTICS": distilbert_score_optics,
            "SentenceTransformer_OPTICS": sentence_transformer_score_optics,
        }
        sorted_rankings = sorted(all_rankings.items(), key=lambda item: item[1], reverse=True)
        print("\nClustering Rankings (by Silhouette Score):")
        for method, score in sorted_rankings:
            print(f"{method}: {score}")

        # Optionally, compute additional ranking metrics for DBSCAN as an example.
        rank_metrics_distilbert = self.rank_clusterings(distilbert_embeddings, lines, distilbert_labels_dbscan)
        rank_metrics_sentence = self.rank_clusterings(sentence_transformer_embeddings, lines, sentence_transformer_labels_dbscan)
        print("\nAdditional Ranking Metrics (using DBSCAN results):")
        print(f"DistilBERT DBSCAN: {rank_metrics_distilbert}")
        print(f"SentenceTransformer DBSCAN: {rank_metrics_sentence}")

