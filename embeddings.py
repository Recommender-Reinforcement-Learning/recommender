import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Read in files
ratings = pd.read_csv(
    '/home/gbz6qn/Documents/MSDS/DS7540/project/ml-32m/ratings.csv',
    dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32},
    parse_dates=['timestamp'],
)
movies = pd.read_csv('/home/gbz6qn/Documents/MSDS/DS7540/project/ml-32m/movies.csv')

# sort by user and timestamp of rating
ratings = ratings.sort_values(['userId', 'timestamp'])

# Create sequences
#user_sequences = ratings.groupby('userId').apply(lambda x: x[['movieId', 'rating']]).values.tolist()

# Map userId and movieId to 0-based integer indices
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()

user_index = {uid: idx for idx, uid in enumerate(user_ids)}
movie_index = {mid: idx for idx, mid in enumerate(movie_ids)}

# Build sparse rating matrix (users x movies)
row = ratings['userId'].map(user_index)
col = ratings['movieId'].map(movie_index)
data = ratings['rating']

rating_matrix = csr_matrix((data, (row, col)), shape=(len(user_ids), len(movie_ids)))

# Initialize SVD to reduce dimensionality to 50 latent features, can change this
svd = TruncatedSVD(n_components=50)

# User embeddings: (num_users x 50)
user_embeddings = svd.fit_transform(rating_matrix)

# Movie embeddings: (num_movies x 50)
movie_embeddings = svd.components_.T

np.save('/home/gbz6qn/Documents/MSDS/DS7540/project/recommender/movie_embeddings.npy', movie_embeddings)
np.save('/home/gbz6qn/Documents/MSDS/DS7540/project/recommender/user_embeddings.npy', user_embeddings)