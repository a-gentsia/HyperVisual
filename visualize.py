import pinecone
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class VectorSpaceAnalyzer:
    def __init__(self, api_key, environment, index_name):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.pinecone = self.initialize_pinecone()
        self.index = self.pinecone.Index(self.index_name)
        self.all_vectors = None

    def initialize_pinecone(self):
        pinecone.init(api_key=self.api_key, environment=self.environment)
        return pinecone

    def fetch_vectors(self, batch_size=1000):
        if self.all_vectors is None:
            index_info = self.index.describe()
            total_vector_count = index_info["totalVector"]
            vectors = []
            for i in range(0, total_vector_count, batch_size):
                ids = [str(j) for j in range(i, min(i+batch_size, total_vector_count))]
                response = self.index.fetch(ids)
                batch_vectors = [v['values'] for v in response['vectors'].values()]
                vectors.extend(batch_vectors)
            self.all_vectors = np.array(vectors)
        return self.all_vectors

    def find_optimal_clusters(self, max_clusters=20):
        vectors = self.fetch_vectors()
        sse = []
        for k in range(1, max_clusters):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(vectors)
            sse.append(kmeans.inertia_)

        # Determine the "elbow" point
        elbows = np.diff(sse, 2)
        optimal_clusters = np.argmax(elbows) + 2

        return optimal_clusters

    def analyze_vector_space(self):
        vectors = self.fetch_vectors()
        optimal_clusters = self.find_optimal_clusters()
        print(f"Optimal number of clusters: {optimal_clusters}")

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(vectors)

        # Perform PCA for visualization
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)

        # Visualize the clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=cluster_labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title('PCA visualization of vector clusters')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.savefig('vector_clusters.png')
        plt.close()

        # Find centroids in original space
        centroids = kmeans.cluster_centers_

        # Function to find nearest vector to a given point
        def find_nearest_vector(point, vectors):
            distances = np.linalg.norm(vectors - point, axis=1)
            return np.argmin(distances)

        # Find nearest actual vectors to centroids
        nearest_to_centroids = [find_nearest_vector(centroid, vectors) for centroid in centroids]

        # Find outliers (e.g., points far from their cluster center)
        distances_to_centroid = np.min(kmeans.transform(vectors), axis=1)
        outlier_threshold = np.percentile(distances_to_centroid, 95)  # Top 5% as outliers
        outlier_indices = np.where(distances_to_centroid > outlier_threshold)[0]

        return {
            "cluster_labels": cluster_labels,
            "centroids": centroids,
            "nearest_to_centroids": nearest_to_centroids,
            "outlier_indices": outlier_indices
        }

    def generate_sample_queries(self, analysis_results, num_samples=5):
        cluster_labels, centroids, nearest_to_centroids, outlier_indices = (
            analysis_results["cluster_labels"],
            analysis_results["centroids"],
            analysis_results["nearest_to_centroids"],
            analysis_results["outlier_indices"]
        )

        # Generate sample queries for centroids
        centroid_queries = []
        for i, nearest_idx in enumerate(nearest_to_centroids):
            cluster_id = cluster_labels[nearest_idx]
            query = f"This is a sample query for the vectors in cluster {cluster_id}."
            centroid_queries.append(query)

        # Generate sample queries for outliers
        outlier_queries = []
        for i, outlier_idx in enumerate(outlier_indices[:num_samples]):
            query = f"This is a sample query for an outlier vector."
            outlier_queries.append(query)

        return {
            "centroid_queries": centroid_queries,
            "outlier_queries": outlier_queries
        }

# Example usage
api_key = "YOUR_API_KEY"
environment = "YOUR_ENVIRONMENT"
index_name = "YOUR_INDEX_NAME"

analyzer = VectorSpaceAnalyzer(api_key, environment, index_name)
analysis_results = analyzer.analyze_vector_space()
sample_queries = analyzer.generate_sample_queries(analysis_results)

print("Centroid sample queries:")
for query in sample_queries["centroid_queries"]:
    print(query)

print("\nOutlier sample queries:")
for query in sample_queries["outlier_queries"]:
    print(query)