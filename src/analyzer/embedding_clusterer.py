# embedding_clusterer.py
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import threading
from collections import defaultdict

class EmbeddingClusterer:
    """
    Performs clustering on embeddings and visualizes the results.
    This class is designed to be self-contained with minimal dependencies
    on other parts of the pipeline.
    """
    def __init__(self, radius=0.5, min_samples=2):
        """
        Initialize the clusterer with parameters.
        
        Args:
            radius: The maximum distance between two samples to be considered in the same cluster
            min_samples: The minimum number of samples in a neighborhood to form a cluster
        """
        self.radius = radius
        self.min_samples = min_samples
        self.clustered_data = None
        self.cluster_centers = None
        self.n_clusters = 0
        self.reduced_embeddings = None
        self.reduced_centers = None
        self.pca = None

    def cluster(self, embeddings_data):
        """
        Perform clustering on the provided embeddings data.
        
        Args:
            embeddings_data: List of dictionaries containing 'embedding', 'lyric', and 'combined_text'
            
        Returns:
            The processed embeddings with cluster assignments added
        """
        if not embeddings_data:
            print("No embeddings to cluster.")
            return embeddings_data

        # Extract embeddings and other data
        embeddings = np.array([data['embedding'] for data in embeddings_data])
        
        # Perform DBSCAN clustering
        db = DBSCAN(eps=self.radius, min_samples=self.min_samples, metric='cosine')
        cluster_labels = db.fit_predict(embeddings)
        
        # Count the number of clusters (excluding noise points labeled as -1)
        self.n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        print(f"Clustering complete. Found {self.n_clusters} clusters.")
        
        # Create a list of data points with their cluster assignments
        self.clustered_data = []
        for i, data in enumerate(embeddings_data):
            # Create a copy of the original data to avoid modifying it
            clustered_item = data.copy()
            clustered_item['cluster'] = int(cluster_labels[i])
            self.clustered_data.append(clustered_item)
        
        # Calculate cluster centers for each valid cluster
        self.cluster_centers = {}
        for cluster_id in set(cluster_labels):
            if cluster_id != -1:  # Skip noise points
                # Get all embeddings for this cluster
                cluster_embeddings = [d['embedding'] for d in self.clustered_data if d['cluster'] == cluster_id]
                # Calculate centroid (mean of all embeddings in the cluster)
                self.cluster_centers[cluster_id] = np.mean(cluster_embeddings, axis=0)
        
        # Prepare reduced embeddings for visualization
        self._prepare_visualization_data(embeddings)
        
        return self.clustered_data

    def _prepare_visualization_data(self, embeddings):
        """Prepare data for visualization using PCA."""
        # Apply PCA for visualization
        self.pca = PCA(n_components=2)
        self.reduced_embeddings = self.pca.fit_transform(embeddings)
        
        # Also transform cluster centers
        if self.cluster_centers:
            center_embeddings = np.array(list(self.cluster_centers.values()))
            self.reduced_centers = self.pca.transform(center_embeddings)

    def optimize_clusters(self, iterations=5, move_factor=0.2):
        """
        Optimize clusters by moving points closer to their cluster centers.
        
        Args:
            iterations: Number of optimization iterations
            move_factor: How much to move points toward their centers (0-1)
            
        Returns:
            The optimized embeddings data
        """
        if not self.clustered_data or not self.cluster_centers:
            print("No clustered data available for optimization.")
            return self.clustered_data
        
        print(f"Optimizing clusters with {iterations} iterations...")
        
        for iteration in range(iterations):
            # For each data point that belongs to a cluster
            for i, data in enumerate(self.clustered_data):
                cluster_id = data['cluster']
                embedding = data['embedding']
                
                # Skip noise points
                if cluster_id == -1:
                    continue
                
                # Get the cluster center
                center = self.cluster_centers[cluster_id]
                
                # Move the embedding towards the cluster center
                optimized_embedding = embedding + move_factor * (center - embedding)
                
                # Normalize to maintain the embedding's scale
                optimized_embedding = optimized_embedding / np.linalg.norm(optimized_embedding)
                
                # Update the embedding
                self.clustered_data[i]['embedding'] = optimized_embedding
            
            # Re-calculate cluster centers
            for cluster_id in self.cluster_centers:
                cluster_embeddings = [d['embedding'] for d in self.clustered_data if d['cluster'] == cluster_id]
                if cluster_embeddings:  # Make sure the cluster still has points
                    self.cluster_centers[cluster_id] = np.mean(cluster_embeddings, axis=0)
        
        print("Cluster optimization complete.")
        
        # Re-prepare visualization data with optimized embeddings
        embeddings = np.array([data['embedding'] for data in self.clustered_data])
        self._prepare_visualization_data(embeddings)
        
        return self.clustered_data

    def visualize_clusters(self, output_file='clustered_embeddings.png'):
        """
        Create a visualization of the clustered embeddings.
        
        Args:
            output_file: Path to save the visualization
        """
        if not self.clustered_data:
            print("No clustered data to visualize.")
            return
        
        # Extract data for plotting
        cluster_labels = [data['cluster'] for data in self.clustered_data]
        texts = [data['combined_text'] for data in self.clustered_data]
        lyrics = [data['lyric'] for data in self.clustered_data]
        
        # Create a colormap that distinguishes clusters clearly
        # -1 (noise) will be black
        unique_labels = set(cluster_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        color_map = {}
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                color_map[label] = [0, 0, 0, 1]  # Black for noise
            else:
                color_map[label] = colors[i]
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot each point colored by its cluster
        for i, (x, y, cluster, text) in enumerate(zip(
                self.reduced_embeddings[:, 0],
                self.reduced_embeddings[:, 1],
                cluster_labels,
                texts)):
            plt.scatter(x, y, color=color_map[cluster], alpha=0.7, s=50)
            plt.annotate(text, (x, y), fontsize=8, alpha=0.7)
        
        # Plot cluster centers
        if self.reduced_centers is not None:
            cluster_ids = list(self.cluster_centers.keys())
            for i, (x, y, cluster) in enumerate(zip(
                    self.reduced_centers[:, 0],
                    self.reduced_centers[:, 1],
                    cluster_ids)):
                plt.scatter(x, y, color=color_map[cluster], marker='X', s=100, 
                           edgecolor='black', linewidth=1.5)
                plt.annotate(f'Cluster {cluster}', (x, y), fontsize=10, 
                           fontweight='bold', ha='center', va='bottom')
        
        plt.title(f'Embedding Clusters (Found {self.n_clusters} clusters)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(output_file)
        plt.close()
        print(f"Saved cluster visualization to {output_file}")
        
        # Generate a second visualization with lyrics
        lyrics_output_file = output_file.replace('.png', '_lyrics.png')
        self._visualize_lyrics_clusters(cluster_labels, lyrics, color_map, lyrics_output_file)

    def _visualize_lyrics_clusters(self, cluster_labels, lyrics, color_map, output_file):
        """
        Create a visualization of the clustered embeddings with lyrics as labels.
        
        Args:
            cluster_labels: List of cluster labels for each point
            lyrics: List of lyrics for each point
            color_map: Dictionary mapping cluster IDs to colors
            output_file: Path to save the visualization
        """
        plt.figure(figsize=(12, 10))
        
        # Plot each point colored by its cluster
        for i, (x, y, cluster, lyric) in enumerate(zip(
                self.reduced_embeddings[:, 0],
                self.reduced_embeddings[:, 1],
                cluster_labels,
                lyrics)):
            plt.scatter(x, y, color=color_map[cluster], alpha=0.7, s=50)
            plt.annotate(lyric, (x, y), fontsize=8, alpha=0.7)
        
        # Plot cluster centers
        if self.reduced_centers is not None:
            cluster_ids = list(self.cluster_centers.keys())
            for i, (x, y, cluster) in enumerate(zip(
                    self.reduced_centers[:, 0],
                    self.reduced_centers[:, 1],
                    cluster_ids)):
                plt.scatter(x, y, color=color_map[cluster], marker='X', s=100, 
                           edgecolor='black', linewidth=1.5)
                plt.annotate(f'Cluster {cluster}', (x, y), fontsize=10, 
                           fontweight='bold', ha='center', va='bottom')
        
        plt.title(f'Lyrics Clusters (Found {self.n_clusters} clusters)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(output_file)
        plt.close()
        print(f"Saved lyrics cluster visualization to {output_file}")
    
    def get_cluster_statistics(self):
        """
        Calculate and return statistics about the clusters.
        
        Returns:
            Dictionary with cluster statistics
        """
        if not self.clustered_data:
            return None
        
        total_points = len(self.clustered_data)
        noise_points = sum(1 for d in self.clustered_data if d['cluster'] == -1)
        clustered_points = total_points - noise_points
        
        # Count points in each cluster
        cluster_sizes = defaultdict(int)
        for data in self.clustered_data:
            cluster_sizes[data['cluster']] += 1
        
        # Remove noise cluster from statistics
        if -1 in cluster_sizes:
            del cluster_sizes[-1]
        
        # Calculate statistics
        stats = {
            'total_points': total_points,
            'clustered_points': clustered_points,
            'noise_points': noise_points,
            'noise_percentage': (noise_points / total_points) * 100 if total_points > 0 else 0,
            'cluster_count': self.n_clusters,
            'avg_cluster_size': sum(cluster_sizes.values()) / len(cluster_sizes) if cluster_sizes else 0,
            'cluster_sizes': dict(cluster_sizes)
        }
        
        return stats
