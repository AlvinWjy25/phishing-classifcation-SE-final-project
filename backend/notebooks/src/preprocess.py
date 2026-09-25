import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class URLEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.model_ = None

    def fit(self, X, y=None):
        self.model_ = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        check_is_fitted(self, ['model_'])

        if self.model_ is None:
            self.model_ = SentenceTransformer(self.model_name)

        if isinstance(X, np.ndarray):
            X = X.flatten().tolist()

        if isinstance(X, str):
            X = [X]

        print("⚙️ Encoding URL...")
        return self.model_.encode(X)

    def __sklearn_is_fitted__(self):
        return hasattr(self, 'model_') and self.model_ is not None