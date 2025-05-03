import torch
import random
import numpy as np
import pandas as pd
import dill
from DataLoader import load_data
from Agent import Actor, Critic, sample_action
from Train import train

#----Set Random Seeds for Reproducibility----
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

#----Load Data----
movies, ratings, user_embeddings, movie_embeddings, genre_embeddings_matrix, movieId_to_genre_index, movieId_to_index, index_to_movieId, unique_user_ids, userId_to_index, index_to_userId = load_data()

# --- Dimensions ---
EMBED_DIM = user_embeddings.shape[1]
GENRE_EMBED_DIM = genre_embeddings_matrix.shape[1]
STATE_DIM = EMBED_DIM + GENRE_EMBED_DIM
REWARD_FUNC = lambda x: (((x-2.75)*5)**3)/(10**3)
GAMMA = 0.50
NUM_EPISODES = 100_000

#----Initialize Actor and Critic----
actor = Actor(input_dim=STATE_DIM, output_dim=len(movie_embeddings))
critic = Critic(input_dim=STATE_DIM)

#----Optimizers----
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

#----Start Training----
reward_history, actor_losses, critic_losses, hit_rates, eval_scores = train(actor, actor_optimizer, critic,
                                                                            critic_optimizer, num_episodes=NUM_EPISODES, 
                                                                            reward_func=REWARD_FUNC,
                                                                            gamma=GAMMA, print_interval=500, eval_interval=2500)

with open('reward_history2.dill', 'wb') as f:
    dill.dump(reward_history, f)
    
with open('actor_losses2.dill', 'wb') as f:
    dill.dump(actor_losses, f)
    
with open('critic_losses2.dill', 'wb') as f:
    dill.dump(critic_losses, f)
    
with open('hit_rates2.dill', 'wb') as f:
    dill.dump(hit_rates, f)
    
with open('eval_scores2.dill', 'wb') as f:
    dill.dump(eval_scores, f)