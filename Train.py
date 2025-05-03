import torch
import numpy as np
import random
from DataLoader import load_data
from Agent import Actor, Critic, sample_action
from Environment import get_state, evaluate_policy

#----Load Data----
movies, ratings, user_embeddings, movie_embeddings, genre_embeddings_matrix, movieId_to_genre_index, movieId_to_index, index_to_movieId, unique_user_ids, userId_to_index, index_to_userId = load_data()

# --- Trackers ---
reward_history = []
actor_losses = []
critic_losses = []
hit_rates = []
eval_scores = []

def train(actor, actor_optim, critic, critic_optim, num_episodes, reward_func, gamma=0.99, print_interval=500, eval_interval=2500):
    for episode in range(num_episodes):
        user_id = random.randint(0, user_embeddings.shape[0] - 1)
        rated = ratings[ratings['userId'] == user_id]
        rated_movie_ids = rated['movieId'].unique()
        available_movies = [movieId_to_index[mid] for mid in rated_movie_ids if mid in movieId_to_index]

        if len(available_movies) == 0:
            continue

        history = []
        total_reward = 0
        correct_recommendations = 0

        for t in range(10):
            state = get_state(user_id, history, user_embeddings, movie_embeddings, movies, movieId_to_genre_index, genre_embeddings_matrix)
            probs = actor(state)

            action_index = sample_action(probs, available_movies)
            movie_id = index_to_movieId[action_index]

            reward_row = rated[rated['movieId'] == movie_id]
            raw_reward = reward_row['rating'].item()
            reward = reward_func(raw_reward)
            if raw_reward >= 4:
                correct_recommendations += 1

            next_state = get_state(user_id, history + [action_index], user_embeddings, movie_embeddings, movies, movieId_to_genre_index, genre_embeddings_matrix)
            value = critic(state)
            next_value = critic(next_state).detach()
            td_target = reward + gamma * next_value
            td_error = td_target - value

            # Critic update
            critic_loss = td_error.pow(2).mean()
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()
            critic_losses.append(critic_loss.item())

            # Actor update
            log_prob = torch.log(probs[action_index])
            actor_loss = -log_prob * td_error.detach()
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()
            actor_losses.append(actor_loss.item())

            history.append(action_index)
            total_reward += reward

        reward_history.append(total_reward)
        hit_rates.append(correct_recommendations / 10)

        if episode % print_interval == 0:
            avg_reward = np.mean(reward_history[-print_interval:])
            avg_hit = np.mean(hit_rates[-print_interval:])
            print(f"[Ep {episode}] Avg Reward: {avg_reward:.3f} | Hit Rate: {avg_hit:.2f}")

        if episode % eval_interval == 0:
            test_users = random.sample(list(ratings['userId'].unique()), 100)
            eval_reward = evaluate_policy(actor, user_embeddings, movie_embeddings, ratings, test_users, movies, movieId_to_genre_index, genre_embeddings_matrix, reward_func)
            eval_scores.append((episode, eval_reward))
            print(f"[Eval @ Ep {episode}] Avg Test Reward: {eval_reward:.3f}")
    
    torch.save(actor.state_dict(), "trained_actor2.pt")
    
    return reward_history, actor_losses, critic_losses, hit_rates, eval_scores
