import pandas as pd
import numpy as np
import random
import torch
from sklearn.decomposition import TruncatedSVD

#---Load Files---
# Paths are hard coded here, not idea but easier to implement
def load_data():
    #----Handle Ratings----
    ratings = ratings = pd.read_csv('/home/gbz6qn/Documents/MSDS/DS7540/project/ml-32m/ratings.csv')
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings = ratings.sort_values(['userId', 'timestamp'])
    
    user_embeddings = np.load('/home/gbz6qn/Documents/MSDS/DS7540/project/IL/user_embeddings.npy')
    
    #----Handle Movies----
    movies = pd.read_csv('/home/gbz6qn/Documents/MSDS/DS7540/project/IL/movies_one_hot_genres.csv', index_col='Unnamed: 0')
    
    new_index = movies['movieId']
    movie_embeddings = movies[['Action', 'Adventure',
       'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
       'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery',
       'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']].copy()
    movie_embeddings['movieId'] = movies['movieId']
    
    #----Initialize SVD for Genre Embeddings----
    n_genre_components = 10  # can experiment with this value
    genre_svd = TruncatedSVD(n_components=n_genre_components)
    genre_embeddings_matrix = genre_svd.fit_transform(movie_embeddings)

    #----Create mappings----
    movieId_to_genre_index = {row['movieId']: index for index, row in movies[['movieId']].iterrows()}
    movieId_to_index = {movie_id: idx for idx, movie_id in enumerate(movies['movieId'])}
    index_to_movieId = {idx: movie_id for movie_id, idx in movieId_to_index.items()}
    unique_user_ids = ratings['userId'].unique()
    userId_to_index = {uid: idx for idx, uid in enumerate(unique_user_ids)}
    index_to_userId = {idx: uid for uid, idx in userId_to_index.items()}


    return (
        movies, ratings, user_embeddings, movie_embeddings, genre_embeddings_matrix,
        movieId_to_genre_index, movieId_to_index, index_to_movieId, 
        unique_user_ids, userId_to_index, index_to_userId
    )
