import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if sys.platform.startswith('win'):
    try:
        import ctypes
        ctypes.CDLL('libiomp5md.dll')
    except Exception as e:
        print(f"Failed to preload DLL: {e}")

# Set OpenMP environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP to single thread

import faiss
import numpy as np
import pickle

class VectorDatabase:
    def __init__(self, vector_dimension, index_file='db/vector_index.faiss', metadata_file='db/vector_metadata.pkl'):
        self.dimension = vector_dimension
        self.index_file = self.resolve_path(index_file)
        self.metadata_file = self.resolve_path(metadata_file)
        self.index = None
        self.metadata = []
        self.load_or_create_index()
    
    def resolve_path(self, file_path):
        if os.path.isabs(file_path):
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            return file_path
        else:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            return os.path.abspath(file_path)
            
    def load_or_create_index(self):
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"Loaded existing index with {self.index.ntotal} vectors")
            except Exception as e:
                print(f"Error loading existing index: {e}")
                print("Creating new index")
                self.create_new_index()
        else:
            print("Index files not found. Creating new index")
            self.create_new_index()

    def create_new_index(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []

    def add_vector(self, vector, metadata):
        vector = np.array([vector], dtype=np.float32)
        self.index.add(vector)
        self.metadata.append(metadata)

    def save_index(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"Saved index with {self.index.ntotal} vectors")

    def search_similar(self, query_vector, k=5):
        query_vector = np.array([query_vector], dtype=np.float32
        )
        distances, indices = self.index.search(query_vector, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], round(float(distances[0][i]), 6)))
        return results