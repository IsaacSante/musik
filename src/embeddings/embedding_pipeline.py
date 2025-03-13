# src/embeddings/embedding_pipeline.py

import torch
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
from nltk.tokenize import word_tokenize
import string
from collections import defaultdict, Counter
import threading
import queue
import time
import os

class EmbeddingPipeline:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the EmbeddingPipeline with a single efficient model for embedding generation and clustering.
        """
        # Download required NLTK resources
        print("Setting up NLTK resources...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Select device: use MPS if available, otherwise fall back to CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS device for acceleration.")
        else:
            self.device = torch.device("cpu")
            print("MPS not available, using CPU.")
        
        # Initialize just one model - the efficient MiniLM
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Set up queue for lyric processing
        self.lyric_queue = queue.Queue()
        self.processed_lyrics = []
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        
        print("EmbeddingPipeline initialized successfully.")
    
    def enqueue_lyric(self, lyric_data):
        """
        Add a lyric data dictionary to the processing queue
        
        Args:
            lyric_data (dict): Dictionary containing the lyric text and analysis data
        """
        if not isinstance(lyric_data, dict) or 'text' not in lyric_data:
            print("Warning: Invalid lyric data format, must be a dictionary with 'text' key")
            return
            
        self.lyric_queue.put(lyric_data)
        self.processed_lyrics.append(lyric_data)
    
    def _process_queue(self):
        """Background thread to process lyrics in the queue"""
        last_process_time = 0
        min_process_interval = 10  # Process at most every 10 seconds to collect more lyrics
        
        while True:
            current_time = time.time()
            # Check if enough time has passed and we have lyrics to process
            if current_time - last_process_time > min_process_interval and not self.lyric_queue.empty():
                # Get all available lyrics
                lyrics_batch = []
                try:
                    while True:
                        lyric = self.lyric_queue.get_nowait()
                        lyrics_batch.append(lyric)
                except queue.Empty:
                    pass
                
                # Only process if we have at least 3 lyrics for meaningful clustering
                if len(lyrics_batch) >= 3:
                    try:
                        print(f"Processing {len(lyrics_batch)} lyrics...")
                        self.analyze_lyrics([lyric['text'] for lyric in lyrics_batch])
                        last_process_time = time.time()
                    except Exception as e:
                        print(f"Error processing lyrics batch: {str(e)}")
            
            time.sleep(0.5)  # Check for new lyrics every half second
    
    def get_embeddings(self, texts):
        """
        Get embeddings using our single model
        """
        return self.model.encode(texts)
    
    def get_normalized_embeddings(self, texts):
        """
        Get normalized embeddings from the model
        """
        embeddings = self.get_embeddings(texts)
        return normalize(np.array(embeddings))

    def tune_clustering(self, embeddings, lines):
        """
        Find optimal clustering parameters using a streamlined parameter search
        """
        best_score = -1  # Initialize with a low score
        best_params = None
        best_labels = None

        # Streamlined parameter search based on log analysis
        min_samples_values = [2, 3]
        max_eps_values = [1.0, 2.0]
        
        for min_samples in min_samples_values:
            for max_eps in max_eps_values:
                try:
                    optics_model = OPTICS(min_samples=min_samples, max_eps=max_eps, cluster_method='xi')
                    labels = optics_model.fit_predict(embeddings)
                    
                    # Only compute silhouette score if more than one cluster is found
                    if len(np.unique(labels)) > 1 and len(labels) > 1:
                        score = silhouette_score(embeddings, labels)
                    else:
                        score = -2
                    
                    print(f"OPTICS: min_samples={min_samples}, max_eps={max_eps}, Silhouette={score}")
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'min_samples': min_samples, 'max_eps': max_eps}
                        best_labels = labels
                        
                        # Early stopping if we find a good score
                        if score > 0.5:
                            print("Found good clustering solution, stopping search")
                            break
                except Exception as e:
                    print(f"Error with min_samples={min_samples}, max_eps={max_eps}: {e}")

        if best_score > -1:
            print(f"\nBEST OPTICS: {best_params}, Silhouette={best_score}")
        
        # If no good labels were found, try a fallback approach
        if best_labels is None:
            print("No valid clustering found, trying fallback")
            try:
                optics_model = OPTICS(min_samples=2, max_eps=2.0)
                best_labels = optics_model.fit_predict(embeddings)
            except Exception as e:
                print(f"Fallback clustering failed: {e}")
                # Last resort: assign all to one cluster
                best_labels = np.zeros(len(embeddings), dtype=int)
            best_params = {'min_samples': 2, 'max_eps': 2.0}
            best_score = -1
            
        return best_labels, best_params, best_score

    def extract_cluster_labels(self, lines, labels):
        """
        For each cluster (ignoring noise), tokenizes all lines,
        removes stopwords and punctuation, and then assigns the cluster
        a label based on the most common token.
        """
        try:
            # Get stopwords
            stop_words = set(stopwords.words('english'))
            
            cluster_to_text = defaultdict(list)
            for line, label in zip(lines, labels):
                if label != -1:
                    cluster_to_text[label].append(line)
            
            cluster_labels = {}
            for label, texts in cluster_to_text.items():
                combined_text = " ".join(texts).lower()
                # Use simple splitting instead of word_tokenize to avoid NLTK issues
                tokens = combined_text.split()
                # Clean tokens
                tokens = [token for token in tokens 
                         if token not in stop_words and 
                         token not in string.punctuation and
                         len(token) > 1]
                
                if tokens:
                    most_common_token, _ = Counter(tokens).most_common(1)[0]
                else:
                    most_common_token = f"Cluster {label}"
                cluster_labels[label] = most_common_token.capitalize()
            return cluster_labels
        except Exception as e:
            print(f"Error extracting cluster labels: {e}")
            # Fallback to simple numbering
            return {label: f"Cluster {label}" for label in set(labels) if label != -1}

    def merge_similar_clusters(self, embeddings, labels, distance_threshold=0.5):
        """
        Merges clusters that have centroids closer than a specified distance threshold.
        Noise points (labeled as -1) remain unchanged.
        """
        try:
            labels = np.array(labels)
            unique_labels = [label for label in np.unique(labels) if label != -1]
            
            # If there are no valid clusters or only one cluster, return the original labels
            if len(unique_labels) <= 1:
                return labels
                
            centroids = {}

            # Compute centroids for each cluster
            for label in unique_labels:
                centroids[label] = embeddings[labels == label].mean(axis=0)
            
            merged_map = {}
            new_cluster_id = 0

            # Compare each cluster centroid with others
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

            # Assign new labels while keeping noise (-1) unchanged
            new_labels = np.array([merged_map[label] if label in merged_map else -1 for label in labels])
            return new_labels
        except Exception as e:
            print(f"Error merging clusters: {e}")
            return labels

    def visualize_embeddings(self, embeddings, lines, labels, title="Lyric Embeddings"):
        """
        Uses t-SNE to reduce the embeddings to 2D and then visualizes
        them along with their cluster labels. The legend is moved to the right side.
        """
        try:
            if len(lines) < 2:
                print("Not enough data points for visualization (need at least 2)")
                return
                
            emb_array = np.array(embeddings)
            # Set perplexity based on the number of data points
            perplexity = min(5, len(lines) - 1)
            
            # Use exact method for small datasets - faster and more stable
            method = 'exact' if len(lines) < 50 else 'barnes_hut'
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, method=method)
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
            
            # Only add text annotations for datasets under 50 points
            if len(lines) < 50:
                for i, line in enumerate(lines):
                    # Truncate long lines for better visualization
                    short_line = line[:20] + "..." if len(line) > 20 else line
                    plt.annotate(short_line, (emb_2d[i, 0], emb_2d[i, 1]), fontsize=7)
                
            plt.title(title)
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.tight_layout()
            plt.savefig("fused_lyric_embeddings.png", dpi=100)  # Lower DPI for faster rendering
            plt.close()
            print("Visualization saved as fused_lyric_embeddings.png")
        except Exception as e:
            print(f"Error visualizing embeddings: {e}")
    
    def analyze_lyrics(self, lyrics, merge_clusters=True, merge_distance_threshold=0.5):
        """
        Main analysis method: computes embeddings, clusters them,
        optionally merges similar clusters, and visualizes the results.
        
        Args:
            lyrics: Either a string (to be split by newlines) or a list of strings
            merge_clusters: Whether to merge similar clusters
            merge_distance_threshold: Threshold for merging clusters
            
        Returns:
            List of cluster labels
        """
        try:
            # Handle both string input and list input
            if isinstance(lyrics, str):
                lines = [line.strip() for line in lyrics.split("\n") if line.strip()]
            else:
                lines = lyrics
                
            if not lines:
                print("No lyrics provided for analysis.")
                return
            
            if len(lines) < 3:
                print(f"Not enough lyrics ({len(lines)}) for clustering. Need at least 3.")
                return
            
            print("Computing embeddings...")
            embeddings = self.get_normalized_embeddings(lines)
            
            print("Clustering embeddings using OPTICS...")
            labels, params, score = self.tune_clustering(embeddings, lines)
            if labels is None:
                print("Clustering did not produce valid labels; assigning default cluster (0) to all lines.")
                labels = np.zeros(len(lines), dtype=int)
            print("Best clustering parameters:", params)
            if score > -2:
                print("Silhouette score:", score)
            
            # Only merge clusters if we have enough data and merge_clusters is True
            if merge_clusters and len(lines) > 5:
                print("Merging similar clusters...")
                labels = self.merge_similar_clusters(embeddings, labels, distance_threshold=merge_distance_threshold)
            
            self.visualize_embeddings(embeddings, lines, labels, title="Lyric Embeddings")
            return labels
        except Exception as e:
            print(f"Error in analyze_lyrics: {e}")
            return None
