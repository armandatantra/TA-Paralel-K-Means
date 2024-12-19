import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Membaca data saham
df = pd.read_csv('c:/Users/USER/split_acak_1.csv')

# Mengambil hanya fitur 'high' dan 'low' untuk proses clustering
fitur_saham = df[['high', 'low']]
x_array = np.array(fitur_saham)

#Normalisasi data menggunakan MinMaxScaler agar data berada pada skala 0-1
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)

from numba import prange 
from joblib import Parallel, delayed
import time

# Perhitungan jarak secara manual
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def assign_labels(data_chunk, centroids):
    n_samples = data_chunk.shape[0]
    k = centroids.shape[0]
    labels = np.zeros(n_samples, dtype=np.int32)
    
    for i in prange(n_samples):
        min_dist = np.inf
        for j in range(k):
            dist = euclidean_distance(data_chunk[i], centroids[j])
            if dist < min_dist:
                min_dist = dist
                labels[i] = j

    return labels

def update_centroids(data, labels, k):
    n_samples, n_features = data.shape
    centroids = np.zeros((k, n_features))
    counts = np.zeros(k, dtype=np.int32)
    
    for i in range(n_samples):
        centroids[labels[i]] += data[i]
        counts[labels[i]] += 1
    
    for j in range(k):
        if counts[j] > 0:
            centroids[j] /= counts[j]
    
    return centroids

def run_kmeans_parallel(data, k, n_jobs, max_iters=300):
    np.random.seed(0)  # centroid konsisten

    # Pilih centroid awal
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    start_time = time.time()
    
    for _ in range(max_iters):
        # Split data menjadi chunk berdasarkan jumlah core
        chunks = np.array_split(data, n_jobs)

        # Parallel job untuk assign label
        results = Parallel(n_jobs=n_jobs)(delayed(assign_labels)(chunk, centroids) for chunk in chunks)
        
        # Gabungkan hasil dari semua chunk
        labels = np.concatenate(results)

        # Update centroid secara global, menggunakan semua data
        new_centroids = update_centroids(data, labels, k)

        # Jika centroid tidak berubah, hentikan iterasi
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids

    end_time = time.time()
    execution_time = end_time - start_time
    
    return labels, centroids, execution_time

# Penggunaan
if __name__ == "__main__":
    data = x_scaled
    k = 2
    
    for n_jobs in [3]: # core yang digunakan
        labels, centroids, execution_time = run_kmeans_parallel(data, k, n_jobs)
        print(f"Core: {n_jobs}")
        print(f"Waktu Eksekusi: {execution_time:.3f} detik")
        df['Cluster'] = labels
        print(f"Centroids:\n{centroids}")
        #print(df)
        
