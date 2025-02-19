# File: src/analyzer/lyrics_analyzer.py
from transformers import DistilBertTokenizer, DistilBertModel
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
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
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(lines) - 1)) #Added perplexity
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
            all_words = ' '.join(cluster_lines).lower()  # Combine all lines into one string

            # Tokenize the words and remove punctuation
            words = [word for word in all_words.split() if word not in string.punctuation]
            from collections import Counter
            word_counts = Counter(words)
            most_common_words = [word for word, count in word_counts.most_common(3)]  # Use top 3 words for name

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

            # Sum TF-IDF scores across all documents in the cluster
            term_sums = tfidf_matrix.sum(axis=0)
            term_scores = [(word, term_sums[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            term_scores = sorted(term_scores, key=lambda x: x[1], reverse=True)

            # Get top 5 keywords
            top_keywords = [word for word, score in term_scores[:5]]
            cluster_keywords[cluster_id] = top_keywords
            print(f"Cluster {cluster_id} Keywords: {top_keywords}")

        return cluster_keywords

        plt.savefig(f"lyric_embeddings_{model_name}_{method_name}.png")
        plt.close()

    def visualize_all_clusterings(self, embeddings, lines, model_name):

        """
        Generates and saves t-SNE plots for multiple clustering methods applied to the given embeddings.
        The image filenames include both the model name and the clustering method.
        """
        emb_array = np.array(embeddings)
        # Added perplexity
        tsne = TSNE(n_components=2, random_state=42, perplexity= min(10, len(lines) - 1))
        emb_2d = tsne.fit_transform(emb_array)

        # Define various clustering methods with desired parameters.
        clustering_methods = {
            "KMeans_k5": KMeans(n_clusters=5, random_state=42, n_init=10),  # Increased k, added n_init
            "KMeans_k7": KMeans(n_clusters=7, random_state=42, n_init=10),  # Increased k, added n_init
            "Agglomerative_ward_k5": AgglomerativeClustering(n_clusters=5, linkage='ward'),
            "Agglomerative_complete_k5": AgglomerativeClustering(n_clusters=5, linkage='complete'),
            "Agglomerative_average_k5": AgglomerativeClustering(n_clusters=5, linkage='average'),
            "Agglomerative_single_k5": AgglomerativeClustering(n_clusters=5, linkage='single'),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=2),
            "HDBSCAN_min3": hdbscan.HDBSCAN(min_cluster_size=3),
            "HDBSCAN_min5": hdbscan.HDBSCAN(min_cluster_size=5),
            "GaussianMixture_k5": GaussianMixture(n_components=5, random_state=42),
            "GaussianMixture_k7": GaussianMixture(n_components=7, random_state=42)
        }

        for method_name, model in clustering_methods.items():
            try:  # Add a try-except block in case a clustering method fails
                if method_name.startswith("GaussianMixture"):  # Handle GMM separately
                    labels = model.fit_predict(emb_array)
                else:
                    labels = model.fit_predict(emb_array)

                cluster_names = self.generate_cluster_names(labels, lines)
                cluster_keywords = self.extract_keywords(labels, lines)

                plt.figure(figsize=(8, 6))
                # Prepare scatter plot with labels
                scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="viridis", s=50)

                # Create legend with cluster names and colors
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f"{cluster_names[i]}",
                                              markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=8)
                                  for i in np.unique(labels) if i != -1]
                if -1 in np.unique(labels):
                    legend_elements.append(plt.Line2D([0], [0], marker='x', color='w', label=f"{cluster_names[-1]}", markerfacecolor='gray', markersize=8))
                plt.legend(handles=legend_elements, loc="best")

                plt.title(f"{model_name} - {method_name}")
                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")

                # Annotate each point with its corresponding lyric line (optional)
                for i, txt in enumerate(lines):
                    plt.annotate(txt, (emb_2d[i, 0], emb_2d[i, 1]), fontsize=8)

                # Save the figure with a filename that includes both the model name and clustering method.
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
        # Optionally print HDBSCAN clustering results:
        self.cluster_lyrics(distilbert_embeddings, lines, "DistilBERT")
        # Save single-view t-SNE plot (optional)
        self.visualize_embeddings(distilbert_embeddings, lines, "DistilBERT")
        # Generate multi-method clustering visualizations
        self.visualize_all_clusterings(distilbert_embeddings, lines, "DistilBERT")  # Remove remove_stopwords

        # Process using SentenceTransformer embeddings.
        sentence_transformer_embeddings = [self.get_sentence_transformer_embedding(line) for line in lines]
        self.cluster_lyrics(sentence_transformer_embeddings, lines, "SentenceTransformer")
        self.visualize_embeddings(sentence_transformer_embeddings, lines, "SentenceTransformer")
        self.visualize_all_clusterings(sentence_transformer_embeddings, lines, "SentenceTransformer") # Remove remove_stopwords

