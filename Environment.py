import torch
import random
import numpy as np
import pandas as pd
from DataLoader import load_data
from Agent import Actor, Critic, sample_action

#----Load Data----
movies, ratings, user_embeddings, movie_embeddings, genre_embeddings_matrix, movieId_to_genre_index, movieId_to_index, index_to_movieId, unique_user_ids, userId_to_index, index_to_userId = load_data()

def get_state(user_id, history, user_embeddings, movie_embeddings, movies_df, movieId_to_genre_index, genre_embeddings_matrix):
    if user_id not in userId_to_index:
        raise ValueError(f"user_id {user_id} not in userId_to_index")

    user_index = userId_to_index[user_id]
    user_embed = torch.tensor(user_embeddings[user_index], dtype=torch.float32)
    history_genre_embed = torch.zeros(genre_embeddings_matrix.shape[1], dtype=torch.float32)

    if history:
        history_movie_genre_embeds = []
        for movie_index in history:
            movie_id = index_to_movieId[movie_index]
            if movie_id in movieId_to_genre_index:
                genre_index = movieId_to_genre_index[movie_id]
                if 0 <= genre_index < len(genre_embeddings_matrix):
                    history_movie_genre_embeds.append(
                        torch.tensor(genre_embeddings_matrix[genre_index], dtype=torch.float32)
                    )

        if history_movie_genre_embeds:
            history_genre_embed = torch.stack(history_movie_genre_embeds).mean(dim=0)

    return torch.cat([user_embed, history_genre_embed], dim=0)

def evaluate_policy(actor, user_embeddings, movie_embeddings, ratings, test_users, movies, movieId_to_genre_index, genre_embeddings_matrix, reward_func, k=1):
    rewards = []
    for user_id in test_users:
        rated = ratings[ratings['userId'] == user_id]
        available = [movieId_to_index[m] for m in rated['movieId'] if m in movieId_to_index]
        if not available:
            continue
        state = get_state(user_id, [], user_embeddings, movie_embeddings, movies, movieId_to_genre_index, genre_embeddings_matrix)
        probs = actor(state)
        action = sample_action(probs, available)
        movie_id = index_to_movieId[action]
        reward_row = rated[rated['movieId'] == movie_id]
        if not reward_row.empty:
            raw_reward = reward_row['rating'].item()
            reward = reward_func(raw_reward)
            rewards.append(reward)        
    return np.mean(rewards) if rewards else 0.0

