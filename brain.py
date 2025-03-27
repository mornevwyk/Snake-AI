import torch
from torch import nn
import numpy as np
from collections import deque
import settings
import random

class NeuralNet(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super().__init__()
        self.neural_net = nn.Sequential(
            nn.Linear(inputSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, 3)
        )

    def forward(self, x):
        x = self.neural_net(x)
        return x
    
class Brain:
    def __init__(self, input, hidden):
        self.NN : NeuralNet = NeuralNet(input, hidden)
        self.targetNN : NeuralNet = NeuralNet(input, hidden)
        self.memory = deque(maxlen = 200000)
        self.gamma = settings.GAMMA
        self.tau = settings.TAU
        self.epsilon = settings.EPSILON
        self.epsilon_decay = settings.EPSILON_DECAY
        self.epsilon_min = settings.EPSILON_MIN
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.NN.parameters(), settings.LR)
        self.count = 0
        self.checkpoint_path = settings.checkpoint_path
        self.load_model()

    def store_memory(self, memory):
        self.memory.append(memory)

    def predict(self, input):
        with torch.no_grad():
            input = input.unsqueeze(0)
            return self.NN(input).squeeze(0)
        
    def save_model(self):
        """Save model and optimizer state, but not the target network."""
        checkpoint = {
            "model_state": self.NN.state_dict(),  # Save the online network only
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon  # Save epsilon from SnakeAI
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Model saved to {self.checkpoint_path}")
    
    def load_model(self):
        """Load model and optimizer state if checkpoint exists."""
        try:
            checkpoint = torch.load(self.checkpoint_path)
            self.NN.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.epsilon = checkpoint.get("epsilon", 1.0)  # Default epsilon if not saved
            print(f"Model loaded from {self.checkpoint_path}")
        except FileNotFoundError:
            print("No saved model found, starting fresh.")
        
    def train(self, batch_size = 32):

        if len(self.memory) < batch_size:
            return
        
        samples = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*samples)

        state = torch.stack(state)
        action = torch.stack(action).argmax(dim=1)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.stack(next_state)
        done = torch.tensor(done, dtype=torch.float32)

        Q_current = self.NN(state).gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad(): 
            best_action = self.NN(next_state).argmax(1)
            Q_target = reward + self.gamma * self.targetNN(next_state).gather(1, best_action.unsqueeze(1)).squeeze(1) * (1 - done)    

        loss = self.loss_fn(Q_current, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.count += 1

        if self.count >= 500:
            self.update_target_network()
            self.count = 0

        if self.count >= 1000:
            self.save_model()
        

    def update_target_network(self):
        """ Soft update target network (smoother updates) """
        for target_param, local_param in zip(self.targetNN.parameters(), self.NN.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
        