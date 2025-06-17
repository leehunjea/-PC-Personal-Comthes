from sklearn.cluster import KMeans
import numpy as np

class ColorExtractor:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    def extract_dominant_colors(self, image: np.ndarray) -> np.ndarray:
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(pixels)
        return kmeans.cluster_centers_.astype(int)
