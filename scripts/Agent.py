import random
import torch
import torch.nn as nn
import torch.nn.functional as F

#----Actor Class----
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.output(x), dim=-1)

#----Critic Class----
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)

#----Sample From Available Movies----
def sample_action(probs, available_movies):
    available_movies_tensor = torch.tensor(available_movies, dtype=torch.long)
    probs_filtered = probs[available_movies_tensor]

    if probs_filtered.sum().item() == 0 or torch.any(torch.isnan(probs_filtered)):
        idx = random.randint(0, len(available_movies) - 1)
    else:
        probs_filtered /= probs_filtered.sum()
        idx = torch.multinomial(probs_filtered, 1).item()

    return available_movies[idx]

