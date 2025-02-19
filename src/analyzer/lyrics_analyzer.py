from transformers import DistilBertTokenizer, DistilBertModel
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import hdbscan
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import string
from collections import defaultdict
from sklearn.preprocessing import normalize

# Download stopwords if you haven't already
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

class LyricsAnalyzer:
    def __init__(self):
        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_distilbert_embedding(self, text):
        inputs = self.distilbert_tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.distilbert_model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        embedding = last_hidden_state.mean(dim=1)
        return embedding.squeeze().numpy()

    def get_sentence_transformer_embedding(self, text):
        return self.sentence_transformer_model.encode(text)

    def cluster_lyrics(self, embeddings, lines, model_name, min_cluster_size=3):
        emb_array = np.array(embeddings)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(emb_array)

        print(f"\nCluster Results for {model_name}:")
        for i, line in enumerate(lines):
            print(f"Cluster {labels[i]}: {line}")
        return labels

    def visualize_embeddings(self, embeddings, lines, model_name):
        emb_array = np.array(embeddings)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(lines) - 1))
        emb_2d = tsne.fit_transform(emb_array)

        plt.figure(figsize=(8, 6))
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1])

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
            if cluster_id == -1:  # Handle noise cluster
                cluster_names[cluster_id] = "Noise"
                continue

            cluster_lines = [lines[i] for i, label in enumerate(labels) if label == cluster_id]
            all_words = ' '.join(cluster_lines).lower()

            # Tokenize the words and remove punctuation
            words = [word for word in all_words.split() if word not in string.punctuation]
            from collections import Counter
            word_counts = Counter(words)
            most_common_words = [word for word, count in word_counts.most_common(3)]
            cluster_names[cluster_id] = ", ".join(most_common_words)

        return cluster_names

    def extract_keywords(self, labels, lines):
        """Extracts keywords for each cluster using TF-IDF."""
        cluster_keywords = {}
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Handle noise cluster
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

    def calculate_embedding_dispersion(self, embeddings):
        """Calculates how dispersed the embeddings are from the center."""
        center = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - center, axis=1)
        mean_distance = np.mean(distances)
        return mean_distance

    def calculate_color_coherence(self, embeddings, labels):
        """Calculates the coherence of colors based on cluster labels."""
        cluster_centers = defaultdict(list)
        for i, label in enumerate(labels):
            cluster_centers[label].append(embeddings[i])

        coherence_score = 0
        for label, embeddings_in_cluster in cluster_centers.items():
            if label == -1:
                continue
            embeddings_in_cluster = np.array(embeddings_in_cluster)
            center = np.mean(embeddings_in_cluster, axis=0)
            distances = np.linalg.norm(embeddings_in_cluster - center, axis=1)
            coherence_score += np.mean(distances)

        num_clusters = len(cluster_centers) - (1 if -1 in cluster_centers else 0)
        if num_clusters > 0:
            coherence_score /= num_clusters
        else:
            coherence_score = float('inf')

        return coherence_score

    def analyze_largest_cluster_spread(self, emb_2d, labels):
        """Calculates a score penalizing spread from the center of the largest cluster, focusing on dimension 2."""
        cluster_sizes = {}
        for label in np.unique(labels):
            cluster_sizes[label] = np.sum(labels == label)

        largest_cluster_id = max(cluster_sizes, key=cluster_sizes.get)
        if largest_cluster_id == -1 or cluster_sizes[largest_cluster_id] <= 1:
            return 0

        largest_cluster_points = emb_2d[labels == largest_cluster_id]
        if len(largest_cluster_points) == 0:
            return 0

        cluster_center = np.mean(largest_cluster_points, axis=0)
        distances = np.abs(largest_cluster_points[:, 1] - cluster_center[1])
        max_distance = np.max(distances)
        penalty = -max_distance
        return penalty

    def rank_clusterings(self, embeddings, lines):
        """Ranks clustering methods based on normalized dispersion, coherence, and largest cluster spread."""
        emb_array = np.array(embeddings)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(lines) - 1))
        emb_2d = tsne.fit_transform(emb_array)

        clustering_methods = {
            # "KMeans_k5": KMeans(n_clusters=5, random_state=42, n_init=10),
            # "KMeans_k7": KMeans(n_clusters=7, random_state=42, n_init=10),
            # "Agglomerative_ward_k5": AgglomerativeClustering(n_clusters=5, linkage='ward'),
            # "Agglomerative_complete_k5": AgglomerativeClustering(n_clusters=5, linkage='complete'),
            # "Agglomerative_average_k5": AgglomerativeClustering(n_clusters=5, linkage='average'),
            # "Agglomerative_single_k5": AgglomerativeClustering(n_clusters=5, linkage='single'),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=2),
            "HDBSCAN_min3": hdbscan.HDBSCAN(min_cluster_size=3),
            "HDBSCAN_min5": hdbscan.HDBSCAN(min_cluster_size=5),
            # "GaussianMixture_k5": GaussianMixture(n_components=5, random_state=42),
            # "GaussianMixture_k7": GaussianMixture(n_components=7, random_state=42),
            "OPTICS": OPTICS(min_samples=2, max_eps=0.5, cluster_method='xi')
        }

        # Containers for raw scores
        dispersion_scores = {}
        coherence_scores = {}
        spread_scores = {}

        for method_name, model in clustering_methods.items():
            try:
                if method_name.startswith("GaussianMixture"):
                    labels = model.fit_predict(emb_array)
                else:
                    labels = model.fit_predict(emb_array)

                # Note: dispersion does not depend on labels and is computed from emb_2d.
                dispersion_scores[method_name] = self.calculate_embedding_dispersion(emb_2d)
                coherence_scores[method_name] = self.calculate_color_coherence(emb_2d, labels)
                spread_scores[method_name] = self.analyze_largest_cluster_spread(emb_2d, labels)
            except Exception as e:
                print(f"Error during clustering with {method_name}: {e}")
                dispersion_scores[method_name] = np.nan
                coherence_scores[method_name] = np.nan
                spread_scores[method_name] = np.nan

        # Helper: min-max normalization for a dictionary of scores.
        def normalize_dict(scores_dict):
            values = np.array(list(scores_dict.values()), dtype=float)
            min_val, max_val = np.nanmin(values), np.nanmax(values)
            # Handle the case where all values are the same.
            if max_val - min_val == 0:
                return {k: 0.0 for k in scores_dict}
            return {k: (v - min_val) / (max_val - min_val) if not np.isnan(v) else np.nan 
                    for k, v in scores_dict.items()}


        norm_dispersion = normalize_dict(dispersion_scores)
        norm_coherence = normalize_dict(coherence_scores)
        norm_spread = normalize_dict(spread_scores)

        rankings = {}
        # Combine normalized metrics (weights can be adjusted)
        for method in clustering_methods.keys():
            rankings[method] = norm_spread.get(method, 0) - norm_dispersion.get(method, 0) - norm_coherence.get(method, 0)

        sorted_rankings = sorted(rankings.items(), key=lambda item: item[1], reverse=True)
        return sorted_rankings

    def visualize_all_clusterings(self, embeddings, lines, model_name, all_rankings):
        """
        Generates and saves t-SNE plots for multiple clustering methods applied to the given embeddings.
        The image filenames include both the model name and the clustering method.
        """
        emb_array = np.array(embeddings)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(lines) - 1))
        emb_2d = tsne.fit_transform(emb_array)

        clustering_methods = {
            # "KMeans_k5": KMeans(n_clusters=5, random_state=42, n_init=10),
            # "KMeans_k7": KMeans(n_clusters=7, random_state=42, n_init=10),
            # "Agglomerative_ward_k5": AgglomerativeClustering(n_clusters=5, linkage='ward'),
            # "Agglomerative_complete_k5": AgglomerativeClustering(n_clusters=5, linkage='complete'),
            # "Agglomerative_average_k5": AgglomerativeClustering(n_clusters=5, linkage='average'),
            # "Agglomerative_single_k5": AgglomerativeClustering(n_clusters=5, linkage='single'),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=2),
            "HDBSCAN_min3": hdbscan.HDBSCAN(min_cluster_size=3),
            # "HDBSCAN_min5": hdbscan.HDBSCAN(min_cluster_size=5),
            # "GaussianMixture_k5": GaussianMixture(n_components=5, random_state=42),
            # "GaussianMixture_k7": GaussianMixture(n_components=7, random_state=42),
            "OPTICS": OPTICS(min_samples=2, max_eps=0.5, cluster_method='xi')
        }

        for method_name, model in clustering_methods.items():
            try:
                if method_name.startswith("GaussianMixture"):
                    labels = model.fit_predict(emb_array)
                else:
                    labels = model.fit_predict(emb_array)

                cluster_names = self.generate_cluster_names(labels, lines)
                cluster_keywords = self.extract_keywords(labels, lines)

                plt.figure(figsize=(8, 6))
                scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="viridis", s=50)
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                              label=f"{cluster_names[i]}",
                                              markerfacecolor=scatter.cmap(scatter.norm(i)),
                                              markersize=8)
                                   for i in np.unique(labels) if i != -1]
                if -1 in np.unique(labels):
                    legend_elements.append(
                        plt.Line2D([0], [0], marker='x', color='w',
                                   label=f"{cluster_names[-1]}",
                                   markerfacecolor='gray', markersize=8))
                plt.legend(handles=legend_elements, loc="best")
                ranking_score = next((score for method, score in all_rankings if method == method_name), "N/A")
                plt.title(f"{model_name} - {method_name} (Score: {ranking_score:.2f})")
                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")

                for i, txt in enumerate(lines):
                    plt.annotate(txt, (emb_2d[i, 0], emb_2d[i, 1]), fontsize=8)

                plt.savefig(f"lyric_embeddings_{model_name}_{method_name}.png")
                plt.close()
            except Exception as e:
                print(f"Error during clustering with {method_name}: {e}")

    def analyze_lyrics(self, lyrics):
        lines = [line.strip() for line in lyrics.split("\n") if line.strip()]
        lines = [line for line in lines if line]

        if not lines:
            print("No lyrics to analyze.")
            return

        # Process using DistilBERT embeddings.
        distilbert_embeddings = [self.get_distilbert_embedding(line) for line in lines]
        # Normalize the embeddings so that each vector is L2-normalized.
        distilbert_embeddings = normalize(np.array(distilbert_embeddings))

        print("DistilBERT Clustering:")
        self.cluster_lyrics(distilbert_embeddings, lines, "DistilBERT")
        self.visualize_embeddings(distilbert_embeddings, lines, "DistilBERT")
        distilbert_rankings = self.rank_clusterings(distilbert_embeddings, lines)

        # Process using SentenceTransformer embeddings.
        sentence_transformer_embeddings = [self.get_sentence_transformer_embedding(line) for line in lines]
        sentence_transformer_embeddings = normalize(np.array(sentence_transformer_embeddings))

        print("\nSentenceTransformer Clustering:")
        self.cluster_lyrics(sentence_transformer_embeddings, lines, "SentenceTransformer")
        self.visualize_embeddings(sentence_transformer_embeddings, lines, "SentenceTransformer")
        sentence_transformer_rankings = self.rank_clusterings(sentence_transformer_embeddings, lines)

        # Combine and print rankings.
        all_rankings = {f"DistilBERT_{k}": v for k, v in dict(distilbert_rankings).items()}
        all_rankings.update({f"SentenceTransformer_{k}": v for k, v in dict(sentence_transformer_rankings).items()})
        sorted_all_rankings = sorted(all_rankings.items(), key=lambda item: item[1], reverse=True)
        rankings_string = "\n".join([f"{method}: {score}" for method, score in sorted_all_rankings])
        print("\nAll Clustering Rankings:\n" + rankings_string)

        # Visualize clusterings for each method.
        self.visualize_all_clusterings(distilbert_embeddings, lines, "DistilBERT", distilbert_rankings)
        self.visualize_all_clusterings(sentence_transformer_embeddings, lines, "SentenceTransformer", sentence_transformer_rankings)
