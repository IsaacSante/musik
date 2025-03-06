from sentence_transformers import SentenceTransformer
from sklearn.cluster import OPTICS, DBSCAN, KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import threading
from typing import Dict, List, Set, Optional
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from threading import Thread
from queue import Queue
import matplotlib
from sklearn.preprocessing import StandardScaler
import os
from difflib import SequenceMatcher
from tqdm import tqdm
matplotlib.use('Agg')  # Use non-interactive backend for server environments
from typing import Dict, List, Set, Optional

class LyricEmbeddingPipeline:
    def __init__(self, 
                 model_name='all-MiniLM-L6-v2', 
                 clustering_method='dbscan',
                 dim_reduction='pca',
                 max_display_lyrics=40):
        """
        Initialize the lyric embedding pipeline with the specified embedding model.
        
        Args:
            model_name (str): Name of the SentenceTransformer model to use
            clustering_method (str): Clustering algorithm to use ('optics', 'dbscan', or 'kmeans')
            dim_reduction (str): Dimensionality reduction method ('pca', 'tsne', or 'umap')
            max_display_lyrics (int): Maximum number of lyrics to display on the detailed plot
        """
        # Initialize embedding model
        self.model = SentenceTransformer(model_name)
        
        # Choose clustering algorithm
        self.clustering_method = clustering_method.lower()
        self.clusterer = self._initialize_clusterer()
        
        # Choose dimensionality reduction technique
        self.dim_reduction = dim_reduction.lower()
        
        # Visualization settings
        self.max_display_lyrics = max_display_lyrics
        
        # Storage for embeddings and data
        self.concepts_data = []  # List of concepts per lyric
        self.subjects_data = []  # List of subjects per lyric
        self.concept_embeddings = []  # List of concept embeddings per lyric
        self.subject_embeddings = []  # List of subject embeddings per lyric
        self.text_labels = []  # Original lyric text
        self.all_data = []  # All lyric data
        
        # Caching
        self.embedding_cache = {}  # Cache for computed embeddings
        self.text_set = set()  # Set of processed text for quick duplicate checking
        
        # Scaler for normalization
        self.scaler = StandardScaler()
        
        # Reduced dimension representations
        self.embedding_2d = None
        
        # Cluster labels
        self.cluster_labels = None
        
        # Visualization
        self.fig = None
        self.ax = None
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Output directory
        self.output_dir = os.getcwd()
        
        print(f"Lyric Embedding Pipeline initialized with model: {model_name}, "
              f"clustering: {clustering_method}, dimensionality reduction: {dim_reduction}")
    
    def _initialize_clusterer(self):
        """Initialize the clustering algorithm based on the selected method."""
        if self.clustering_method == 'optics':
            return OPTICS(min_samples=2, xi=0.05, min_cluster_size=2, metric='cosine')
        elif self.clustering_method == 'dbscan':
            return DBSCAN(eps=0.5, min_samples=2, metric='cosine')
        elif self.clustering_method == 'kmeans':
            return KMeans(n_clusters=5, n_init=10)  # Starts with 5 clusters and will adapt
        else:
            # Default to DBSCAN
            print(f"Unknown clustering method: {self.clustering_method}. Using DBSCAN instead.")
            return DBSCAN(eps=0.5, min_samples=2, metric='cosine')
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity between two strings.
        
        Args:
            text1: First string
            text2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        return SequenceMatcher(None, text1, text2).ratio()
        
    def add_lyric(self, lyric_data: Dict):
        """
        Add a new lyric to the embedding pipeline and update clustering.
        
        Args:
            lyric_data: Dictionary containing 'text', 'concepts', and 'subject' keys.
        """
        # Extract data
        text = lyric_data.get('text', '')
        concepts = lyric_data.get('concepts', [])
        subjects = lyric_data.get('subject', [])
        
        # Skip if missing data
        if not text or not concepts or not subjects:
            print("Skipping lyric with missing data")
            return
            
        # Check for exact duplicates using set for O(1) lookup
        if text in self.text_set:
            print(f"Skipping duplicate lyric: '{text[:30]}{'...' if len(text) > 30 else ''}'")
            return
            
        # Check for near-duplicates (optional) - this is more expensive
        if len(self.text_labels) > 20:  # Only check when we have enough data to worry about
            for existing_text in self.text_labels[-10:]:  # Only check against recent lyrics
                if self._similarity_score(text, existing_text) > 0.85:
                    print(f"Skipping near-duplicate lyric: '{text[:30]}{'...' if len(text) > 30 else ''}'")
                    return
        
        # Create embeddings
        concept_embeddings = self._create_embeddings(concepts)
        subject_embeddings = self._create_embeddings(subjects)
        
        # Calculate average embeddings for concepts and subjects
        avg_concept_embedding = np.mean(concept_embeddings, axis=0) if len(concept_embeddings) > 0 else np.zeros(self.model.get_sentence_embedding_dimension())
        avg_subject_embedding = np.mean(subject_embeddings, axis=0) if len(subject_embeddings) > 0 else np.zeros(self.model.get_sentence_embedding_dimension())
        
        # Acquire lock for thread safety when updating shared data
        with self.lock:
            # Store the data
            self.concepts_data.append(concepts)
            self.subjects_data.append(subjects)
            self.concept_embeddings.append(avg_concept_embedding)
            self.subject_embeddings.append(avg_subject_embedding)
            self.text_labels.append(text)
            self.all_data.append(lyric_data)
            self.text_set.add(text)  # Add to our set of processed texts
            
            # Update clustering and visualization
            self._update_clustering()
            self._visualize_embeddings()
            
        print(f"Added lyric: '{text[:30]}{'...' if len(text) > 30 else ''}' with {len(concepts)} concepts and {len(subjects)} subjects")
    
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts, with caching.
        
        Args:
            texts: List of strings to embed.
            
        Returns:
            Array of embedding vectors.
        """
        if not texts:
            return np.array([])
        
        # Use cached embeddings when available
        embeddings = []
        texts_to_embed = []
        indices = []
        
        for i, text in enumerate(texts):
            if text in self.embedding_cache:
                embeddings.append(self.embedding_cache[text])
            else:
                texts_to_embed.append(text)
                indices.append(i)
        
        # Only compute embeddings for texts not in cache
        if texts_to_embed:
            new_embeddings = self.model.encode(texts_to_embed)
            
            # Store in cache and insert at correct positions
            for i, idx in enumerate(indices):
                self.embedding_cache[texts_to_embed[i]] = new_embeddings[i]
                embeddings.insert(idx, new_embeddings[i])
        
        return np.array(embeddings)
    
    def _update_clustering(self):
        """
        Update the clustering based on current embeddings, with efficient handling for large datasets.
        """
        if len(self.concept_embeddings) < 3:
            print("Not enough data for clustering yet (need at least 3 lyrics)")
            return
        
        # Create a combined feature space using both concept and subject embeddings
        combined_features = np.hstack([
            np.vstack(self.concept_embeddings),
            np.vstack(self.subject_embeddings)
        ])
        
        # Normalize the features
        normalized_features = self.scaler.fit_transform(combined_features)
        
        # Adapt parameters for different dataset sizes
        if len(self.text_labels) <= 5:
            # For very small datasets, use more lenient parameters
            if self.clustering_method == 'dbscan':
                self.clusterer = DBSCAN(eps=0.7, min_samples=2, metric='cosine')
            elif self.clustering_method == 'kmeans':
                self.clusterer = KMeans(n_clusters=min(2, len(self.text_labels)), n_init=10)
                
        elif len(self.text_labels) <= 20:
            # For small datasets
            if self.clustering_method == 'dbscan':
                self.clusterer = DBSCAN(eps=0.6, min_samples=2, metric='cosine')
            elif self.clustering_method == 'kmeans':
                self.clusterer = KMeans(n_clusters=min(3, len(self.text_labels) - 1), n_init=10)
                
        elif len(self.text_labels) > 100:
            # For large datasets
            if self.clustering_method == 'kmeans':
                # Adjust number of clusters based on data size
                n_clusters = min(max(5, len(self.text_labels) // 10), 20)
                self.clusterer = MiniBatchKMeans(n_clusters=n_clusters, 
                                                batch_size=50, 
                                                n_init=3)
            elif self.clustering_method == 'dbscan':
                # Adjust parameters for larger datasets
                self.clusterer = DBSCAN(eps=0.6, min_samples=3, metric='cosine', n_jobs=-1)
        else:
            # For medium datasets, adapt KMeans
            if self.clustering_method == 'kmeans':
                n_clusters = min(max(3, len(self.text_labels) // 3), len(self.text_labels) - 1)
                self.clusterer = KMeans(n_clusters=n_clusters, n_init=10)
        
        # Run clustering
        self.cluster_labels = self.clusterer.fit_predict(normalized_features)
        
        # Perform dimensionality reduction
        if self.dim_reduction == 'tsne' and len(normalized_features) >= 5:
            perplexity = min(30, max(5, len(normalized_features) // 3))
            tsne = TSNE(n_components=2, 
                        perplexity=perplexity,
                        n_iter=1000, 
                        random_state=42, 
                        metric='cosine')
            self.embedding_2d = tsne.fit_transform(normalized_features)
        elif self.dim_reduction == 'umap' and len(normalized_features) >= 5:
            try:
                # Correct import for UMAP
                from umap import UMAP
                n_neighbors = min(15, max(2, len(normalized_features) // 2))
                reducer = UMAP(n_neighbors=n_neighbors,
                              min_dist=0.1, 
                              n_components=2, 
                              metric='cosine',
                              random_state=42)
                self.embedding_2d = reducer.fit_transform(normalized_features)
            except (ImportError, AttributeError) as e:
                print(f"UMAP not available: {e}. Falling back to PCA.")
                self.dim_reduction = 'pca'  # Fall back to PCA
                pca = PCA(n_components=2)
                self.embedding_2d = pca.fit_transform(normalized_features)
        else:  # PCA as default
            pca = PCA(n_components=2)
            self.embedding_2d = pca.fit_transform(normalized_features)
            
        # Count clusters
        if self.cluster_labels is not None:
            unique_labels = set(self.cluster_labels)
            n_clusters = len(unique_labels) - (1 if -1 in self.cluster_labels else 0)
            print(f"Updated clustering: {n_clusters} clusters found among {len(self.text_labels)} lyrics")
    
    def _visualize_embeddings(self):
        """
        Visualize the embeddings in a 2D space and save to file, with subsampling for large datasets.
        """
        if self.embedding_2d is None:
            print("No projections available yet for visualization")
            return
        
        # Create or clear the plot
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(18, 15))  # Larger figure
        else:
            self.ax.clear()
        
        # Always create overview plot if we have enough data (3+ lyrics)
        if len(self.text_labels) >= 3:
            self._create_overview_plot()
        
        # Determine whether to subsample for the detailed plot
        if len(self.text_labels) > self.max_display_lyrics:
            # Subsample points for detailed plot
            indices_to_plot = self._get_representative_samples()
            
            # Use only the subsampled data
            plot_data = self.embedding_2d[indices_to_plot]
            plot_labels = [self.cluster_labels[i] for i in indices_to_plot]
            plot_texts = [self.text_labels[i] for i in indices_to_plot]
            plot_concepts = [self.concepts_data[i] for i in indices_to_plot]
            plot_subjects = [self.subjects_data[i] for i in indices_to_plot]
            
            print(f"Subsampling {len(indices_to_plot)} points from {len(self.text_labels)} for clearer visualization")
        else:
            # Use all data
            plot_data = self.embedding_2d
            plot_labels = self.cluster_labels
            plot_texts = self.text_labels
            plot_concepts = self.concepts_data
            plot_subjects = self.subjects_data
        
        # Add jitter to prevent exact overlaps
        jitter = np.random.normal(0, 0.02, plot_data.shape)
        plot_data = plot_data + jitter
        
        # Plotting
        unique_labels = sorted(set(plot_labels))
        colors = cm.rainbow(np.linspace(0, 1, max(len(unique_labels), 1)))
        
        # For each cluster, plot with a different color
        for i, cluster_id in enumerate(unique_labels):
            if cluster_id == -1:
                cluster_color = 'k'
                cluster_label = 'Noise'
            else:
                cluster_color = colors[i % len(colors)]
                cluster_label = f'Cluster {cluster_id + 1}'
            
            # Get indices of points in this cluster
            cluster_indices = [j for j, label in enumerate(plot_labels) if label == cluster_id]
            
            # Plot the points
            if cluster_indices:  # Ensure we have points
                self.ax.scatter(
                    [plot_data[j, 0] for j in cluster_indices], 
                    [plot_data[j, 1] for j in cluster_indices],
                    s=80,
                    color=cluster_color,
                    alpha=0.7,
                    label=cluster_label
                )
                
                # Add text annotations
                for j in cluster_indices:
                    text = plot_texts[j]
                    # Truncate long texts and add ellipsis
                    short_text = text[:30] + ('...' if len(text) > 30 else '')
                    
                    # Add concepts and subjects as additional info
                    concept_str = ', '.join(plot_concepts[j][:2])
                    subject_str = ', '.join(plot_subjects[j][:2])
                    annotation = f"{short_text}\n[{concept_str}] [{subject_str}]"
                    
                    self.ax.annotate(
                        annotation,
                        xy=(plot_data[j, 0], plot_data[j, 1]),
                        xytext=(5, 5),  # Small offset
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                    )
        
        # Add labels and title
        method_name = {
            'tsne': 't-SNE', 
            'umap': 'UMAP', 
            'pca': 'PCA'
        }.get(self.dim_reduction, self.dim_reduction.upper())
        
        cluster_name = {
            'optics': 'OPTICS',
            'dbscan': 'DBSCAN',
            'kmeans': 'K-Means'
        }.get(self.clustering_method, self.clustering_method.upper())
        
        self.ax.set_xlabel(f'{method_name} Dimension 1')
        self.ax.set_ylabel(f'{method_name} Dimension 2')
        
        if len(self.text_labels) > self.max_display_lyrics:
            sample_note = f" (showing {len(indices_to_plot)} representative lyrics)"
        else:
            sample_note = ""
            
        self.ax.set_title(f'Lyric Embeddings Using {method_name} + {cluster_name} (n={len(self.text_labels)}){sample_note}')
        
        # Add legend if we have multiple clusters
        if len(unique_labels) > 1:
            self.ax.legend(loc='upper right')
        
        # Adjust limits with margins
        x_min, x_max = np.min(plot_data[:, 0]), np.max(plot_data[:, 0])
        y_min, y_max = np.min(plot_data[:, 1]), np.max(plot_data[:, 1])
        margin = 0.15  # 15% margin
        x_margin = (x_max - x_min) * margin
        y_margin = (y_max - y_min) * margin
        self.ax.set_xlim(x_min - x_margin, x_max + x_margin)
        self.ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Use explicit figure size and subplots_adjust instead of tight_layout
        self.fig.set_size_inches(18, 15)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        # Save with bbox_inches='tight' to fix layout issues
        output_path = os.path.join(self.output_dir, 'lyric_embeddings.png')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5, dpi=150)
        
        # Save cluster summary if we have at least 3 lyrics (changed from 5)
        if len(self.text_labels) >= 3:
            self._create_cluster_summary()
        
        print(f"Updated visualization saved to '{output_path}'")
    
    def _get_representative_samples(self) -> List[int]:
        """
        Get representative samples from each cluster for visualization.
        
        Returns:
            List of indices to plot
        """
        unique_labels = sorted(set(self.cluster_labels))
        indices_to_plot = []
        
        # First pass: get centroids and important points from each cluster
        for label in unique_labels:
            if label == -1:  # Handle noise separately
                continue
                
            # Get indices of points in this cluster
            cluster_indices = np.where(self.cluster_labels == label)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # Get cluster center
            cluster_embeddings = self.embedding_2d[cluster_indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find point closest to centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            centroid_idx = cluster_indices[np.argmin(distances)]
            indices_to_plot.append(centroid_idx)
            
            # Get 1-2 more representative points if cluster is large enough
            if len(cluster_indices) > 3:
                # Get points farthest from each other for diversity
                remaining = set(cluster_indices) - {centroid_idx}
                if remaining:
                    # Pick 2 more diverse examples if available
                    additional = np.random.choice(list(remaining), 
                                                size=min(2, len(remaining)), 
                                                replace=False)
                    indices_to_plot.extend(additional)
        
        # Second pass: add some noise points if we have room
        if -1 in unique_labels:
            noise_indices = np.where(self.cluster_labels == -1)[0]
            remaining_slots = min(self.max_display_lyrics - len(indices_to_plot), len(noise_indices))
            
            if remaining_slots > 0 and len(noise_indices) > 0:
                noise_samples = np.random.choice(noise_indices, size=remaining_slots, replace=False)
                indices_to_plot.extend(noise_samples)
        
        # Third pass: if we still have room, add more points from the largest clusters
        if len(indices_to_plot) < self.max_display_lyrics:
            # Count points per cluster
            cluster_sizes = {}
            for label in unique_labels:
                if label == -1:  # Skip noise
                    continue
                cluster_sizes[label] = np.sum(self.cluster_labels == label)
            
            # Sort clusters by size (largest first)
            sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
            
            # Add more points from largest clusters
            remaining_slots = self.max_display_lyrics - len(indices_to_plot)
            for label, size in sorted_clusters:
                if remaining_slots <= 0:
                    break
                    
                # Get indices not already selected
                cluster_indices = np.where(self.cluster_labels == label)[0]
                available = set(cluster_indices) - set(indices_to_plot)
                
                # Add some more if available
                to_add = min(remaining_slots, len(available), size // 3)  # Add up to 1/3 of cluster size
                if to_add > 0 and available:
                    additional = np.random.choice(list(available), size=to_add, replace=False)
                    indices_to_plot.extend(additional)
                    remaining_slots -= to_add
        
        return indices_to_plot
    
    def _create_overview_plot(self):
        """Create a simplified overview plot with all points but minimal annotations."""
        overview_fig, overview_ax = plt.subplots(figsize=(16, 12))
        
        # Plot all points colored by cluster
        unique_labels = sorted(set(self.cluster_labels))
        colors = cm.rainbow(np.linspace(0, 1, max(len(unique_labels), 1)))
        
        # Add jitter to prevent exact overlaps
        jitter = np.random.normal(0, 0.02, self.embedding_2d.shape)
        plot_data = self.embedding_2d + jitter
        
        # First plot all points with smaller markers
        for i, cluster_id in enumerate(unique_labels):
            cluster_color = 'gray' if cluster_id == -1 else colors[i % len(colors)]
            cluster_points = np.where(self.cluster_labels == cluster_id)[0]
            
            if len(cluster_points) > 0:
                overview_ax.scatter(
                    plot_data[cluster_points, 0], 
                    plot_data[cluster_points, 1],
                    s=30,  # Smaller points
                    color=cluster_color,
                    alpha=0.6,
                    label=f'Cluster {cluster_id + 1}' if cluster_id != -1 else 'Unclustered'
                )
        
        # Only annotate cluster centroids
        for i, cluster_id in enumerate(unique_labels):
            if cluster_id == -1:
                continue
                
            cluster_points = np.where(self.cluster_labels == cluster_id)[0]
            if len(cluster_points) > 0:
                # Calculate centroid
                centroid_x = np.mean(plot_data[cluster_points, 0])
                centroid_y = np.mean(plot_data[cluster_points, 1])
                
                # Get most common concepts and subjects in this cluster
                all_concepts = []
                for idx in cluster_points:
                    all_concepts.extend(self.concepts_data[idx])
                
                # Count concept frequencies
                concept_counts = {}
                for concept in all_concepts:
                    concept_counts[concept] = concept_counts.get(concept, 0) + 1
                    
                # Get top concepts
                top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                concept_str = ", ".join([c for c, _ in top_concepts])
                
                # Annotate centroid
                overview_ax.annotate(
                    f"Cluster {cluster_id+1}\n({len(cluster_points)} lyrics)\n[{concept_str}]",
                    xy=(centroid_x, centroid_y),
                    fontsize=9,
                    weight='bold',
                    ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
                )
                
                # Draw a circle around the cluster
                if len(cluster_points) > 2:  # Changed from 3 to 2 for smaller datasets
                    # Calculate cluster radius (distance from centroid to farthest point)
                    distances = np.sqrt(np.sum((plot_data[cluster_points] - [centroid_x, centroid_y])**2, axis=1))
                    radius = np.max(distances) * 1.1  # Add 10% margin
                    
                    # Draw circle
                    circle = plt.Circle((centroid_x, centroid_y), radius, 
                                       fill=False, 
                                       color=colors[i % len(colors)], 
                                       linestyle='--',
                                       alpha=0.5)
                    overview_ax.add_patch(circle)
        
        method_name = {
            'tsne': 't-SNE', 
            'umap': 'UMAP', 
            'pca': 'PCA'
        }.get(self.dim_reduction, self.dim_reduction.upper())
        
        # Add titles with smaller dataset size note
        if len(self.text_labels) < 10:
            overview_ax.set_title(f'Overview of All {len(self.text_labels)} Lyrics ({method_name}) - Small Dataset')
        else:
            overview_ax.set_title(f'Overview of All {len(self.text_labels)} Lyrics ({method_name})')
        
        # Add legend for clusters
        if len(unique_labels) > 1:
            legend = overview_ax.legend(loc='upper right', title="Clusters")
            
        # Save with explicit size and bbox_inches='tight'
        output_path = os.path.join(self.output_dir, 'lyric_embedding_overview.png')
        overview_fig.set_size_inches(16, 12)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5, dpi=150)
        plt.close(overview_fig)
        print(f"Saved overview plot to '{output_path}'")

    def _create_cluster_summary(self):
        """Create a summary visualization of clusters with example lyrics from each cluster."""
        if self.cluster_labels is None:
            return
            
        unique_labels = sorted(set(self.cluster_labels))
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        if n_clusters < 1:
            # For small datasets with no clear clusters, create a simplified summary
            self._create_small_dataset_summary()
            return
            
        # Figure height depends on number of clusters and text
        fig_height = max(8, n_clusters * 1.5 + 2)
        fig = plt.figure(figsize=(14, fig_height))
        
        # Use gridspec for more control over layout
        from matplotlib import gridspec
        gs = gridspec.GridSpec(n_clusters + 1, 1)
        
        # Add title
        ax_title = plt.subplot(gs[0])
        ax_title.text(0.5, 0.5, f"Lyric Clusters Summary ({len(self.text_labels)} lyrics in {n_clusters} clusters)",
                     horizontalalignment='center',
                     fontsize=14,
                     fontweight='bold')
        ax_title.axis('off')
        
        # Create cluster summaries
        for i, cluster_id in enumerate([c for c in unique_labels if c != -1]):
            # Get cluster members
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            
            # Skip empty clusters
            if len(cluster_indices) == 0:
                continue
                
            # Create subplot for this cluster
            ax = plt.subplot(gs[i+1])
            
            # Get most common concepts and subjects in this cluster
            all_concepts = [concept for idx in cluster_indices 
                            for concept in self.concepts_data[idx]]
            all_subjects = [subject for idx in cluster_indices 
                           for subject in self.subjects_data[idx]]
            
            # Count frequencies
            concept_counts = {}
            for concept in all_concepts:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
                
            subject_counts = {}
            for subject in all_subjects:
                subject_counts[subject] = subject_counts.get(subject, 0) + 1
            
            # Get top concepts and subjects
            top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top_subjects = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Format as strings
            concept_str = ", ".join([f"{c} ({n})" for c, n in top_concepts])
            subject_str = ", ".join([f"{s} ({n})" for s, n in top_subjects])
            
            # Visualization color
            cluster_color = cm.rainbow(i / max(n_clusters, 1))
            
            # Create cluster summary text
            title = f"Cluster {cluster_id + 1}: {len(cluster_indices)} lyrics"
            ax.text(0.05, 0.95, title, 
                   transform=ax.transAxes,
                   fontsize=12,
                   fontweight='bold',
                   verticalalignment='top')
            
            ax.text(0.05, 0.85, f"Top Concepts: {concept_str}", 
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                   
            ax.text(0.05, 0.70, f"Top Subjects: {subject_str}", 
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add example lyrics (up to 3 or all if <= 3)
            sample_size = min(3, len(cluster_indices))
            sample_indices = np.random.choice(cluster_indices, size=sample_size, replace=False)
            
            example_text = "Example Lyrics:\n"
            for j, idx in enumerate(sample_indices):
                # Truncate long lyrics
                lyric_text = self.text_labels[idx]
                if len(lyric_text) > 50:
                    lyric_text = lyric_text[:50] + "..."
                    
                # Add bullet point
                example_text += f"• \"{lyric_text}\"\n"
            
            ax.text(0.05, 0.55, example_text, 
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=np.array(cluster_color) * 0.3 + 0.7, alpha=0.2))
            
            # Add a colored bar on the left to visually distinguish clusters
            ax.axvline(x=0.01, ymin=0.05, ymax=0.95, color=cluster_color, linewidth=5)
            ax.set_axis_off()
        
        plt.tight_layout(h_pad=1.0)
        output_path = os.path.join(self.output_dir, 'lyric_clusters_summary.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Cluster summary saved to '{output_path}'")
    
    def _create_small_dataset_summary(self):
        """Create a special summary for small datasets with few or no clusters."""
        # Create a simple figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Title
        ax.text(0.5, 0.95, f"Lyric Summary ({len(self.text_labels)} lyrics)",
                horizontalalignment='center',
                fontsize=14,
                fontweight='bold',
                transform=ax.transAxes)
        
        # Get all concepts and subjects
        all_concepts = [concept for concepts in self.concepts_data for concept in concepts]
        all_subjects = [subject for subjects in self.subjects_data for subject in subjects]
        
        # Count frequencies
        concept_counts = {}
        for concept in all_concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
            
        subject_counts = {}
        for subject in all_subjects:
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
        
        # Get top concepts and subjects
        top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        top_subjects = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        
        # Format as strings
        concept_str = ", ".join([f"{c} ({n})" for c, n in top_concepts])
        subject_str = ", ".join([f"{s} ({n})" for s, n in top_subjects])
        
        # Add overall summary
        ax.text(0.5, 0.85, "Overall Themes:",
                horizontalalignment='center',
                fontsize=12,
                transform=ax.transAxes)
                
        ax.text(0.5, 0.78, f"Top Concepts: {concept_str}",
                horizontalalignment='center',
                fontsize=10,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8))
                
        ax.text(0.5, 0.68, f"Top Subjects: {subject_str}",
                horizontalalignment='center',
                fontsize=10,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#f8e8e8', alpha=0.8))
        
        # Add all lyrics
        ypos = 0.58
        ax.text(0.5, ypos, "All Lyrics:",
                horizontalalignment='center',
                fontsize=12,
                transform=ax.transAxes)
        
        # Display lyrics with their concepts
        lyrics_text = ""
        ypos -= 0.05
        for i, (text, concepts) in enumerate(zip(self.text_labels, self.concepts_data)):
            # Format concepts
            concept_text = ", ".join(concepts[:3])
            
            # Add to text
            lyrics_text += f"{i+1}. \"{text}\" [Concepts: {concept_text}]\n\n"
        
        # Add lyrics text in a scrollable box
        lyric_box = ax.text(0.5, 0.35, lyrics_text,
                           horizontalalignment='center',
                           verticalalignment='top',
                           fontsize=9,
                           transform=ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='#f8f8f8', alpha=0.8))
        
        # Hide axes
        ax.axis('off')
        
        # Save the figure
        output_path = os.path.join(self.output_dir, 'lyric_clusters_summary.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Small dataset summary saved to '{output_path}'")

    def get_current_clusters(self):
        """
        Get the current clustering results.
        
        Returns:
            List of tuples containing (text, concepts, subjects, cluster_label)
        """
        if self.cluster_labels is None:
            return []
        
        result = []
        for i, (text, concepts, subjects, cluster) in enumerate(zip(
            self.text_labels,
            self.concepts_data,
            self.subjects_data,
            self.cluster_labels
        )):
            result.append((text, concepts, subjects, cluster))
        
        return result
        
    def clear_data(self):
        """Clear all data to start fresh."""
        self.concepts_data = []
        self.subjects_data = []
        self.concept_embeddings = []
        self.subject_embeddings = []
        self.text_labels = []
        self.all_data = []
        self.text_set = set()
        # Keep the embedding_cache for efficiency
        self.embedding_2d = None
        self.cluster_labels = None
        print("Cleared all lyric data from the pipeline")
