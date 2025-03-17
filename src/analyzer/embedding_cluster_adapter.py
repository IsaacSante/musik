# embedding_cluster_adapter.py
import threading
from src.analyzer.embedding_clusterer import EmbeddingClusterer

class EmbeddingClusterAdapter(threading.Thread):
    """
    Adapter class that connects the EmbeddingProcessor to the EmbeddingClusterer.
    This minimizes direct dependencies between the two classes.
    """
    def __init__(self, embedding_processor, radius=0.5, min_samples=2, output_file='clustered_embeddings.png'):
        super().__init__(daemon=True)
        self.embedding_processor = embedding_processor
        self.output_file = output_file
        self.clusterer = EmbeddingClusterer(radius=radius, min_samples=min_samples)
        
    def run(self):
        """Execute the clustering process in a thread."""
        print("Embedding cluster processing started...")
        
        # Get embeddings from the processor
        embeddings_data = self.embedding_processor.embeddings
        
        # Perform clustering
        clustered_data = self.clusterer.cluster(embeddings_data)
        
        # Optimize clusters
        optimized_data = self.clusterer.optimize_clusters()
        
        # Get and display cluster statistics
        stats = self.clusterer.get_cluster_statistics()
        if stats:
            print("Cluster Statistics:")
            for key, value in stats.items():
                if key != 'cluster_sizes':
                    print(f"  {key}: {value}")
            print("  Cluster sizes:")
            for cluster_id, size in stats['cluster_sizes'].items():
                print(f"    Cluster {cluster_id}: {size} points")
        
        # Visualize the clusters
        self.clusterer.visualize_clusters(self.output_file)
        
        # Update the original embeddings in the processor with the optimized ones
        # This is the only point of contact with the processor after initialization
        for i, data in enumerate(optimized_data):
            if i < len(self.embedding_processor.embeddings):
                self.embedding_processor.embeddings[i]['embedding'] = data['embedding']
                # Add cluster information to the original data
                self.embedding_processor.embeddings[i]['cluster'] = data['cluster']
        
        print("Embedding cluster processing complete.")
