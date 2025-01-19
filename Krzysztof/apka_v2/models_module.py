from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

class MLModels:
    def __init__(self):
        """Initialize models and preprocessors."""
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)  # Example with 2 components; adjust as needed.
        self.kmeans = KMeans(n_clusters=3)  # Example with 3 clusters; adjust as needed.
        self.xgb_classifier = XGBClassifier()

    def scale_data(self, data):
        """Scale the data using StandardScaler.

        Args:
            data (array-like): The data to scale.

        Returns:
            array-like: Scaled data.
        """
        return self.scaler.fit_transform(data)

    def apply_pca(self, data):
        """Apply PCA on the data.

        Args:
            data (array-like): The data to reduce dimensions for.

        Returns:
            array-like: Data transformed by PCA.
        """
        return self.pca.fit_transform(data)

    def cluster_data(self, data):
        """Cluster the data using KMeans.

        Args:
            data (array-like): The data to cluster.

        Returns:
            array-like: Cluster labels.
        """
        return self.kmeans.fit_predict(data)

    def train_xgb(self, X_train, y_train):
        """Train the XGBoost classifier.

        Args:
            X_train (array-like): Training features.
            y_train (array-like): Training labels.
        """
        self.xgb_classifier.fit(X_train, y_train)

    def predict_xgb(self, X_test):
        """Make predictions using the XGBoost classifier.

        Args:
            X_test (array-like): Test features.

        Returns:
            array-like: Predicted labels.
        """
        return self.xgb_classifier.predict(X_test)
