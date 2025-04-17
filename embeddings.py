import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Read in files
ratings = pd.read_csv('/home/gbz6qn/Documents/MSDS/DS7540/project/ml-32m/ratings.csv')
movies = pd.read_csv('/home/gbz6qn/Documents/MSDS/DS7540/project/ml-32m/movies.csv')

# sort by user and timestamp of rating
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings = ratings.sort_values(['userId', 'timestamp'])

# Create sequences
user_sequences = ratings.groupby('userId').apply(lambda x: x[['movieId', 'rating']]).values.tolist()

# Create pivoted matrix of users (sparse matrix)
rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Initialize SVD to reduce dimensionality to 50 latent features, can change this
svd = TruncatedSVD(n_components=50)

# Fit on movies and transform, save to numpy
movie_embeddings = svd.fit_transform(rating_matrix.T)
np.save('/home/gbz6qn/Documents/MSDS/DS7540/project/recommender/movie_embeddings.npy', movie_embeddings)

# Transform user embeddings using fitted SVD, save to numpy
user_embeddings = svd.transform(rating_matrix)
np.save('/home/gbz6qn/Documents/MSDS/DS7540/project/recommender/user_embeddings.npy', movie_embeddings)