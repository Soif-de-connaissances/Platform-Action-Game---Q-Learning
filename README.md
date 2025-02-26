# Reinforcement Learning Platform Game

## Project Overview

This is a platform action game developed with PyGame, combined with a Deep Q-Learning Network (DQN) to enable an intelligent agent to autonomously learn game strategies. The player (or AI agent) needs to move between platforms by jumping while avoiding bullets, ultimately reaching the goal platform.

## Core Features

1. **DQN-based AI Agent**: Uses deep reinforcement learning algorithms to autonomously learn game strategies
2. **Dynamic Difficulty System**: Provides game scenarios with three difficulty levels
3. **Real-time Bullet Dodging**: Bullets fired from multiple directions increase game challenge
4. **Reward Mechanism**: Carefully designed reward function guides AI learning effective strategies
5. **Visualization Interface**: Real-time display of game status, AI decisions, and relevant data

## Technical Implementation

### Deep Q-Learning Network (DQN)

```python
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.LayerNorm(64)
        self.fc4 = nn.Linear(64, action_size)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.LeakyReLU(0.01)
```

Network structure includes:
- 4 fully connected layers
- Layer normalization for improved training stability
- Dropout to prevent overfitting
- LeakyReLU activation function

### Reinforcement Learning Key Components

#### 1. State Space
The game state consists of 9 continuous variables:
- Player position (x, y)
- Player velocity (vx, vy)
- Platform status (whether standing on a platform)
- Distance and height of the next platform
- Relative position of the goal platform

#### 2. Action Space
The agent can perform 3 actions:
- Move left
- Move right
- Jump

#### 3. Reward Function
Carefully designed reward mechanism includes:
- Positive rewards for moving toward the goal
- Rewards for exploring new areas
- Rewards for avoiding bullets
- Rewards for successfully jumping onto platforms
- Additional rewards for reaching higher platforms
- Penalties for staying in place
- Penalties for falling or being hit by bullets
- Large reward for reaching the goal

#### 4. Epsilon-Greedy Exploration Strategy
```python
def act(self, state):
    if np.random.rand() <= self.epsilon:
        return random.randrange(self.action_size)  # Explore
    
    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    act_values = self.model(state)
    return np.argmax(act_values.cpu().data.numpy())  # Exploit
```

- Initial epsilon value set to 1.0, ensuring thorough exploration
- Gradually decays during training (multiplied by 0.998)
- Minimum value set to 0.05, maintaining a certain level of exploration

#### 5. Experience Replay
```python
def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)
    states, actions, rewards, next_states, dones, _ = zip(*minibatch)
    
    # Convert to tensors and calculate Q values
    states = torch.FloatTensor(states).to(self.device)
    actions = torch.LongTensor(actions).to(self.device)
    rewards = torch.FloatTensor(rewards).to(self.device)
    next_states = torch.FloatTensor(next_states).to(self.device)
    dones = torch.FloatTensor(dones).to(self.device)
    
    # Calculate current and target Q values
    current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
    next_q_values = self.target_model(next_states).max(1)[0]
    target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
    
    # Calculate loss and update network
    loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

Uses an experience replay buffer to store and sample past experiences, breaking correlations between samples and improving learning efficiency.

#### 6. Target Network
Uses a separate target network to calculate target Q values, updated every 10 episodes, improving training stability.

## Game Environment

### Difficulty Levels
1. **Easy (0)**: Wide platforms, small gaps
2. **Medium (1)**: Medium-width platforms, moderate gaps
3. **Hard (2)**: Narrow platforms, complex layout, more bullets

### Bullet System
```python
def update_bullets(self):
    current_time = pygame.time.get_ticks()
    
    # Update existing bullet positions
    for bullet in list(self.bullets):
        bullet.update(self.platforms)
        if bullet.is_out_of_screen() or bullet.collides_with_platform(self.platforms):
            self.bullets.remove(bullet)
    
    # Generate new bullets
    if current_time - self.last_spawn_time >= self.bullet_spawn_interval:
        spawn_point = random.choice(self.bullet_spawn_points)
        start_pos, direction = spawn_point
        self.bullets.append(Bullet(start_pos[0], start_pos[1], direction, 5))
        self.last_spawn_time = current_time
```

Bullets are randomly fired from four different directions, increasing game difficulty and complexity.

## Training Process

The training process is divided into multiple stages:
1. **Initial Exploration Stage**: High epsilon value, extensive random exploration
2. **Gradual Learning Stage**: Epsilon gradually decreases, more exploitation of learned knowledge
3. **Increasing Difficulty Stage**: Difficulty increases every 200 episodes
4. **Model Evaluation and Saving**: Regularly evaluates model performance and saves the best model

Training parameters:
- Learning rate: Initially 0.001, later reduced to 0.0001
- Discount factor (gamma): 0.99
- Batch size: 64
- Target network update frequency: Every 10 episodes
- Model saving frequency: Every 50 episodes
- Model evaluation frequency: Every 100 episodes

## Usage Instructions

### Training Mode
Set `train = True` to start training mode:
- Automatically runs 1000 episodes of training
- Dynamically adjusts difficulty
- Periodically saves models
- Accelerated training speed (200 FPS)

### Testing Mode
Set `train = False` to start testing mode:
- Loads pre-trained model (`best_model.pth`)
- Visualization interface shows AI decision-making process
- Displays real-time game status and Q values
- Normal game speed (60 FPS)

### Interface Display
The interface in testing mode includes:
- Game time
- Current difficulty
- Action chosen by AI
- Current reward
- Detailed state information
- Q values for each action

## Project Highlights

1. **Deep Reinforcement Learning Practice**: Complete implementation of DQN algorithm and its key components
2. **Complex Reward Design**: Multi-dimensional reward function guides AI to learn efficient strategies
3. **Dynamic Difficulty System**: Gradually increases difficulty during training, enhancing model generalization
4. **Visualized Decision Process**: Intuitively displays AI decision basis and state evaluation
5. **Stable Training Mechanism**: Target network, experience replay, and other mechanisms ensure training stability

## Future Extensions

1. **Algorithm Improvements**: Implement Double DQN, Dueling DQN, or Prioritized Experience Replay
2. **More Complex Environment**: Add enemies, collectible items, or diverse terrain
3. **Multimodal Input**: Use images as input, combined with CNN processing
4. **Multi-agent System**: Implement multiple AI agents collaborating or competing
5. **Transfer Learning**: Transfer learned strategies to other similar games

## Conclusion

This project demonstrates the application of deep reinforcement learning in game AI, using the DQN algorithm to enable an AI agent to autonomously learn complex platform game strategies. The carefully designed reward function and training mechanisms allow the AI to perform excellently in environments of varying difficulty levels, showcasing the powerful capabilities of reinforcement learning in solving sequential decision problems.
