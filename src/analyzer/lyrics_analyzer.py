import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
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

    def visualize_embeddings(self, embeddings, lines, model_name, labels, nested_labels=None, original_labels=None):
        """Visualizes embeddings using t-SNE with provided labels, including nested visualization."""
        emb_array = np.array(embeddings)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(lines) - 1))
        emb_2d = tsne.fit_transform(emb_array)

        plt.figure(figsize=(8, 6))

        # Plot original clusters as background (optional)
        if original_labels is not None:
            scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=original_labels, cmap="Pastel1", alpha=0.5, label="Original Clusters")

        # Plot nested clusters on top
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="viridis", label="Nested Clusters")


        # Create a unified legend, handling the possibility that only some points have both original
        # and nested cluster labels
        legend_elements = []

        # Get a set of all unique labels used (original and nested)
        all_unique_labels = set(np.unique(labels))
        if original_labels is not None:
            all_unique_labels.update(np.unique(original_labels))
        all_unique_labels = sorted(list(all_unique_labels))

        for i in all_unique_labels:

            marker = 'o' # Default to circle

            if original_labels is not None and i in original_labels:
                label_text = f"Original Cluster {i}"

            if i in labels:
                label_text = f"Nested Cluster {i}"

            if i == -1:
                marker='x' # Noise marker
                label_text = 'Noise'
            else:
                label_text = f"Cluster {i}"

            if i in labels:
                 cmap = scatter.cmap
                 norm = scatter.norm
                 facecolor = cmap(norm(i))

            elif original_labels is not None and i in original_labels:
                 # Assuming pastel colormap exists
                 cmap = plt.cm.Pastel1 # use a default pastel colormap from mpl
                 norm = plt.Normalize(vmin=min(original_labels) if len(lines) > 0 else 0, vmax=max(original_labels) if len(lines) > 0 else 0)
                 facecolor = cmap(norm(i))


            legend_elements.append(plt.Line2D([0], [0], marker=marker,linestyle='', color='w',  # No visible line for legend
                                       label=label_text,
                  	                   markerfacecolor=facecolor,
                    	               markersize=8))



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
        all_results = {
            "DistilBERT_OPTICS": (distilbert_labels_optics, distilbert_embeddings, distilbert_score_optics),
            "SentenceTransformer_OPTICS": (sentence_transformer_labels_optics, sentence_transformer_embeddings, sentence_transformer_score_optics),
        }

        sorted_rankings = sorted(all_results.items(), key=lambda item: item[1][2], reverse=True)
        print("\nClustering Rankings (by Silhouette Score):")
        for method, (labels, embeddings, score) in sorted_rankings:
            print(f"{method}: {score}")

        # Perform nested clustering if applicable
        best_method, (best_labels, best_embeddings, best_score) = sorted_rankings[0]
        self.perform_nested_clustering(best_embeddings, lines, best_labels,best_method)

        # Optionally, compute additional ranking metrics for DBSCAN as an example.
        rank_metrics_distilbert = self.rank_clusterings(distilbert_embeddings, lines, distilbert_labels_optics)
        rank_metrics_sentence = self.rank_clusterings(sentence_transformer_embeddings, lines, sentence_transformer_labels_optics)
        print("\nAdditional Ranking Metrics (using OPTICS results):")
        print(f"DistilBERT OPTICS: {rank_metrics_distilbert}")
        print(f"SentenceTransformer OPTICS: {rank_metrics_sentence}")

    def perform_nested_clustering(self, embeddings, lines, labels, parent_model_name):
        """
        Performs nested clustering within the largest cluster if it meets the size criteria.
        """
        cluster_sizes = {label: np.sum(labels == label) for label in np.unique(labels)}
        if len(cluster_sizes) <= 1:
            print('No clusters found for nest')
            return

        # Remove noise cluster from consideration for size comparison
        if -1 in cluster_sizes:
            del cluster_sizes[-1]

        largest_cluster_id = max(cluster_sizes, key=cluster_sizes.get)
        largest_cluster_size = cluster_sizes[largest_cluster_id]

        # Find the size of the second largest cluster.
        sorted_sizes = sorted(cluster_sizes.values(), reverse=True)
        second_largest_size = sorted_sizes[1] if len(
            sorted_sizes) > 1 else 0  # Handle case where there's only one cluster

        # Calculate the percentage difference in size.
        if second_largest_size > 0:
            size_difference_percentage = (
                                                largest_cluster_size - second_largest_size) / second_largest_size * 100
        else:
            size_difference_percentage = float('inf')  # or a large number if no other clusters exist

        print(f"Largest cluster size: {largest_cluster_size}")
        print(f"Second largest cluster size: {second_largest_size}")
        print(f"Size difference percentage: {size_difference_percentage:.2f}%")

        #Store the original cluster labels for visualization
        original_labels = labels[:]

        # Check if the largest cluster meets the criteria for nested clustering.
        if size_difference_percentage >= SIZE_DIFFERENCE_THRESHOLD:
            print(f"Performing nested clustering on cluster {largest_cluster_id}...")

            # Extract embeddings and lines from the largest cluster.
            largest_cluster_indices = [i for i, label in enumerate(labels) if label == largest_cluster_id]
            largest_cluster_embeddings = embeddings[largest_cluster_indices]
            largest_cluster_lines = [lines[i] for i in largest_cluster_indices]

            # Perform nested clustering (e.g., using OPTICS). Modify params as needed.
            nested_labels, nested_params, nested_score = self.tune_clustering(largest_cluster_embeddings, largest_cluster_lines, "OPTICS")

            if nested_labels is None: #Add this check
                print("Nested clustering failed to produce labels. Skipping further processing.") #inform
                return #exit

            # Adjust nested cluster labels to be unique within the overall clustering.
            nested_labels_adjusted = [
                f"{largest_cluster_id}_{label}" if label != -1 else str(label) for label in nested_labels]

            # Adjust all labels so changes are to all labels, not a subset
            all_labels_adjusted = labels[:]

            # Correctly update nested labels in the main labels array if nested_labels are found and non-empty
            if nested_labels is not None and len(nested_labels) > 0:
                for i, original_index in enumerate(largest_cluster_indices):
                    all_labels_adjusted[original_index] = nested_labels_adjusted[i]

                #Visualize it all together. Now using the correct all_labels_adjusted
                self.visualize_embeddings(embeddings, lines,
                                              f"{parent_model_name}_Nested_Combined_{largest_cluster_id}",
                                              all_labels_adjusted , original_labels=original_labels) #Pass original labels


                self.cluster_lyrics(embeddings, lines,
                                    f"{parent_model_name}_Nested__Combined_{largest_cluster_id}",
                                    all_labels_adjusted)
            else:
                   print("No nested clusters found, skipping visualization")
                   #Output file and name
        else:
           print("Largest cluster does not meet the size criteria for nested clustering.")
