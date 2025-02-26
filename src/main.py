import pygame
import sys
from pygame.math import Vector2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# 训练模式标志（设置为False时将使用学习到的策略而不进行进一步学习）
train = False
# 训练模式文件
def get_train_model_file(episode):
    return f"Model_episode_{episode}.pth"
# 测试模式文件
test_model_file = "best_model.pth"

# 初始化 Pygame
pygame.init()

# 设置窗口
WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 720
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("简单的平台动作闯关游戏")

# 颜色定义
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BROWN = (139, 69, 19)
BLACK = (0, 0, 0)

# 使用DQN的连续状态空间
STATE_SIZE = 9  # 状态维度
ACTION_SIZE = 3  # 动作数量

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

    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.activation(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.learning_rate = 0.001
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.eval()  # 初始化时设置为eval模式

    def remember(self, state, action, reward, next_state, done):
        priority = abs(reward) + 1e-5  # 计算优先级
        self.memory.append((state, action, reward, next_state, done, priority))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones, _ = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# 玩家类
class Player:
    def __init__(self, spawn_platform):
        self.width = 30
        self.height = 50
        self.x = spawn_platform.x + spawn_platform.width // 2 - self.width // 2  # 居中放置
        self.y = spawn_platform.y - self.height  # 放置在平台上方
        self.vel = Vector2(0, 0)
        self.move_speed = 5
        self.jump_strength = -15
        self.gravity = 0.8
        self.max_fall_speed = 15
        self.on_platform = True
        

    def move(self, direction, platforms):
        if direction == "left":
            new_x = self.x - self.move_speed
            # 检查是否超出左边界
            if new_x < 0:
                new_x = 0
            if self.can_move_horizontally(-self.move_speed, platforms):
                self.x = new_x
        elif direction == "right":
            new_x = self.x + self.move_speed
            # 检查是否超出右边界
            if new_x + self.width > WINDOW_WIDTH:
                new_x = WINDOW_WIDTH - self.width
            if self.can_move_horizontally(self.move_speed, platforms):
                self.x = new_x

    def can_move_horizontally(self, dx, platforms):
        future_x = self.x + dx
        for platform in platforms:
            if (future_x + self.width > platform.x and
                future_x < platform.x + platform.width):
                # 检查玩家是否在平台上方或下方
                if self.y + self.height > platform.y and self.y < platform.y + platform.height:
                    # 如果玩家的底部高于平台顶部一定距离，允许移动
                    if self.y + self.height < platform.y + 10:
                        return True
                    return False
        return True

    def jump(self):
        if self.on_platform:  # 只有在平台上才能跳跃
            self.vel.y = self.jump_strength
            self.on_platform = False

    def update(self, platforms):
        self.vel.y += self.gravity
        if self.vel.y > self.max_fall_speed:
            self.vel.y = self.max_fall_speed

        new_y = self.y + self.vel.y
        self.on_platform = False

        for platform in platforms:
            if (self.x < platform.x + platform.width and
                self.x + self.width > platform.x):
                if self.y + self.height <= platform.y and new_y + self.height > platform.y:
                    new_y = platform.y - self.height
                    self.vel.y = 0
                    self.on_platform = True
                    break

        self.y = new_y

    def has_fallen(self):
        return self.y > WINDOW_HEIGHT  # 如果玩家y坐标大于窗口高度，则认为已经掉落

    def draw(self, screen):
        pygame.draw.rect(screen, RED, (self.x, self.y, self.width, self.height))


# 平台类
class Platform:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def draw(self, screen):
        pygame.draw.rect(screen, BROWN, (self.x, self.y, self.width, self.height))

# 子弹类
class Bullet:
    def __init__(self, x, y, direction, speed):
        self.pos = Vector2(x, y)
        self.direction = Vector2(direction)
        self.speed = speed
        self.width = 10
        self.height = 5 if self.direction.x != 0 else 10

    def update(self, platforms):
        self.pos += self.direction * self.speed
        
    def is_out_of_screen(self):
        return (self.pos.x < -self.width or 
                self.pos.x > WINDOW_WIDTH or 
                self.pos.y < -self.height or 
                self.pos.y > WINDOW_HEIGHT)

    def collides_with_platform(self, platforms):
        for platform in platforms:
            if (self.pos.x < platform.x + platform.width and
                self.pos.x + self.width > platform.x and
                self.pos.y < platform.y + platform.height and
                self.pos.y + self.height > platform.y):
                return True
        return False

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, (self.pos.x, self.pos.y, self.width, self.height))

    def collides_with(self, player):
        return (self.pos.x < player.x + player.width and
                self.pos.x + self.width > player.x and
                self.pos.y < player.y + player.height and
                self.pos.y + self.height > player.y)

class GameEnvironment:
    def __init__(self, player, platforms, bullets, end_platform):
        self.player = player
        self.platforms = platforms
        self.bullets = bullets
        self.end_platform = end_platform
        self.previous_distance_to_goal = self._calculate_distance_to_goal()
        self.previous_height = WINDOW_HEIGHT - player.y
        self.was_in_air = False
        self.position_history = []
        self.previous_max_height = WINDOW_HEIGHT - player.y  # 记录最大高度
        self.previous_x = player.x  # 记录上一帧位置

        # 子弹生成配置
        self.bullet_spawn_points = [
            ((WINDOW_WIDTH - 10, WINDOW_HEIGHT - 175), (-1, 0)),
            ((-10, WINDOW_HEIGHT - 325), (1, 0)),
            ((400, -10), (0, 1)),
            ((1000, -10), (0, 1))
        ]
        self.last_spawn_time = pygame.time.get_ticks()
        self.bullet_spawn_interval = 2000

    def update_bullets(self):
        """更新子弹状态并生成新子弹"""
        current_time = pygame.time.get_ticks()
        
        # 更新现有子弹位置
        for bullet in list(self.bullets):
            bullet.update(self.platforms)
            # 检查子弹是否超出屏幕或碰到平台
            if bullet.is_out_of_screen() or bullet.collides_with_platform(self.platforms):
                self.bullets.remove(bullet)
        
        # 生成新子弹
        if current_time - self.last_spawn_time >= self.bullet_spawn_interval:
            spawn_point = random.choice(self.bullet_spawn_points)
            start_pos, direction = spawn_point
            self.bullets.append(Bullet(start_pos[0], start_pos[1], direction, 5))
            self.last_spawn_time = current_time

    def check_bullet_collisions(self):
        """检查子弹与玩家的碰撞"""
        player_rect = pygame.Rect(self.player.x, self.player.y, self.player.width, self.player.height)
        for bullet in self.bullets:
            bullet_rect = pygame.Rect(bullet.pos.x, bullet.pos.y, bullet.width, bullet.height)
            if player_rect.colliderect(bullet_rect):
                return True
        return False

    def calculate_reward(self):
        reward = 0
        
        # 向右移动奖励
        if self.player.x > self.previous_x:
            reward += 0.5

        # 进度奖励
        progress = self.player.x / WINDOW_WIDTH
        reward += progress * 2

        # 探索奖励
        if self._is_exploring_new_area():
            reward += 10

        # 跳跃奖励（增加条件限制）
        if self.player.vel.y < 0 and not self._is_on_platform():  # 只在有效跳跃时奖励
            reward += 10  # 降低奖励值
            
        # 新增水平移动奖励
        if abs(self.player.vel.x) > 2:  # 当有显著水平移动时
            reward += 3 * (self.player.vel.x / self.player.move_speed)  # 方向敏感奖励

        # 高度奖励（提高系数）
        current_height = WINDOW_HEIGHT - self.player.y
        reward += current_height * 0.05

        # 避开子弹的奖励
        for bullet in self.bullets:
            if self._is_close_to_bullet(bullet):
                reward += 2

        # 危险处理奖励/惩罚
        for bullet in self.bullets:
            if bullet.collides_with(self.player):
                reward -= 50
            else:
                bullet_distance = self._calculate_distance_to_bullet(bullet)
                if bullet_distance < 100:
                    reward += 5

        # 平台相关奖励 - 成功到达平台的奖励
        if self._is_on_platform() and self.was_in_air:
            reward += 10
            self.was_in_air = False
            
            # 额外奖励：如果到达了比之前更高的平台
            current_height = WINDOW_HEIGHT - self.player.y
            if current_height > self.previous_max_height:
                reward += 15
                self.previous_max_height = current_height
                
        elif not self._is_on_platform():
            self.was_in_air = True

        # 避免长时间停留在同一位置
        if self._is_stuck_in_place():
            reward -= 5

        # 时间惩罚 - 促使更快行动
        reward -= 0.001

        # 终点方向奖励
        direction_reward = (self.end_platform.x - self.player.x) / WINDOW_WIDTH
        reward += direction_reward * 0.5

        # 更新位置历史
        self.position_history.append((self.player.x, self.player.y))
        if len(self.position_history) > 30:
            self.position_history.pop(0)

        # 更新previous_x
        self.previous_x = self.player.x

        # 获胜或失败奖励
        if self._check_win():
            reward += 2000  # 增加到达终点的奖励
        elif self.player.has_fallen():
            reward -= 50  # 减少掉落的惩罚

        return reward

    def get_state(self):
        player_x = self.player.x / WINDOW_WIDTH
        player_y = self.player.y / WINDOW_HEIGHT
        player_vel_x = (self.player.vel.x + 10) / 20  # 归一化速度到 [0, 1]
        player_vel_y = (self.player.vel.y + 10) / 20
        on_platform = 1 if self.player.on_platform else 0
        
        # 添加最近子弹的相对位置
        nearest_bullet_x = 0
        nearest_bullet_y = 0

        if self.bullets:
            nearest_bullet = min(self.bullets, key=lambda b: ((b.pos.x - self.player.x)**2 + (b.pos.y - self.player.y)**2)**0.5)
            nearest_bullet_x = (nearest_bullet.pos.x - self.player.x + WINDOW_WIDTH) / (2 * WINDOW_WIDTH)
            nearest_bullet_y = (nearest_bullet.pos.y - self.player.y + WINDOW_HEIGHT) / (2 * WINDOW_HEIGHT)

        # 添加终点平台的相对位置
        goal_x = (self.end_platform.x - self.player.x + WINDOW_WIDTH) / (2 * WINDOW_WIDTH)
        goal_y = (self.end_platform.y - self.player.y + WINDOW_HEIGHT) / (2 * WINDOW_HEIGHT)
        
        state = np.array([
            self.player.x / WINDOW_WIDTH,
            self.player.y / WINDOW_HEIGHT,
            self.player.vel.x / self.player.move_speed,
            self.player.vel.y / self.player.jump_strength,
            1.0 if self.player.on_platform else 0.0,
            self._get_next_platform_distance() / WINDOW_WIDTH,
            self._get_next_platform_height() / WINDOW_HEIGHT,
            (self.end_platform.x - self.player.x) / WINDOW_WIDTH,
            (self.end_platform.y - self.player.y) / WINDOW_HEIGHT
        ])
        
        # 确保所有值都在 [0, 1] 范围内，除了 on_platform
        state[:4] = np.clip(state[:4], 0, 1)
        state[5:] = np.clip(state[5:], 0, 1)
        
        return state

    def _get_next_platform_distance(self):
        next_platform = None
        min_distance = float('inf')
        for platform in self.platforms:
            if platform.x > self.player.x:
                dist = platform.x - self.player.x
                if dist < min_distance:
                    min_distance = dist
                    next_platform = platform
        return min_distance if next_platform else WINDOW_WIDTH

    def _get_next_platform_height(self):
        next_platform = None
        min_distance = float('inf')
        for platform in self.platforms:
            if platform.x > self.player.x:
                dist = platform.x - self.player.x
                if dist < min_distance:
                    min_distance = dist
                    next_platform = platform
        return next_platform.y if next_platform else WINDOW_HEIGHT

    def _calculate_distance_to_goal(self):  # 利用欧氏距离进行计算
        return ((self.player.x - self.end_platform.x) ** 2 + 
                (self.player.y - self.end_platform.y) ** 2) ** 0.5

    def _calculate_distance_to_bullet(self, bullet):
        return ((self.player.x - bullet.pos.x) ** 2 + (self.player.y - bullet.pos.y) ** 2) ** 0.5

    def _is_on_platform(self):
        for platform in self.platforms:
            if (self.player.x + self.player.width > platform.x and 
                self.player.x < platform.x + platform.width and 
                abs(self.player.y + self.player.height - platform.y) < 5):
                return True
        return False

    def _is_stuck_in_place(self):
        if len(self.position_history) < 30:
            return False
        start_pos = self.position_history[0]
        end_pos = self.position_history[-1]
        distance_moved = ((end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2) ** 0.5
        return distance_moved < 10  # 假设移动距离小于10像素就认为是卡住了

    def _is_exploring_new_area(self):
        """检查是否在探索新区域"""
        if len(self.position_history) < 2:
            return False
            
        current_pos = self.position_history[-1]
        # 检查是否访问了新的区域
        for pos in self.position_history[:-1]:
            if abs(current_pos[0] - pos[0]) < 50 and abs(current_pos[1] - pos[1]) < 50:
                return False
        return True

    def _check_win(self):
        return (self.player.x >= self.end_platform.x and 
                self.player.x + self.player.width <= self.end_platform.x + self.end_platform.width and 
                abs(self.player.y + self.player.height - self.end_platform.y) < 5)

    def _is_close_to_bullet(self, bullet):
        distance = self._calculate_distance_to_bullet(bullet)
        return distance < 50  # 假设50是一个合适的"接近"距离

def show_win_popup(screen, time_spent):
    popup_width, popup_height = 300, 200
    popup_x = (WINDOW_WIDTH - popup_width) // 2
    popup_y = (WINDOW_HEIGHT - popup_height) // 2
    
    popup_surface = pygame.Surface((popup_width, popup_height))
    popup_surface.fill(WHITE)
    pygame.draw.rect(popup_surface, BLACK, (0, 0, popup_width, popup_height), 2)

    font = pygame.font.Font(None, 36)
    win_text = font.render("You Win!!!", True, BLACK)
    time_text = font.render(f"Time: {time_spent:.2f} seconds", True, BLACK)
    win_rect = win_text.get_rect(center=(popup_width // 2, 40))
    time_rect = time_text.get_rect(center=(popup_width // 2, 80))
    popup_surface.blit(win_text, win_rect)
    popup_surface.blit(time_text, time_rect)

    button_width, button_height = 120, 40
    restart_button = pygame.Rect((popup_width - 2 * button_width - 20) // 2, 120, button_width, button_height)
    quit_button = pygame.Rect(restart_button.right + 20, 120, button_width, button_height)

    pygame.draw.rect(popup_surface, GREEN, restart_button)
    pygame.draw.rect(popup_surface, RED, quit_button)

    restart_text = font.render("Restart", True, BLACK)
    quit_text = font.render("Quit", True, BLACK)
    popup_surface.blit(restart_text, restart_text.get_rect(center=restart_button.center))
    popup_surface.blit(quit_text, quit_text.get_rect(center=quit_button.center))

    screen.blit(popup_surface, (popup_x, popup_y))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                adjusted_pos = (mouse_pos[0] - popup_x, mouse_pos[1] - popup_y)
                if restart_button.collidepoint(adjusted_pos):
                    return "restart"
                elif quit_button.collidepoint(adjusted_pos):
                    return "quit"
        
        pygame.display.flip()  # 确保弹窗持续显示

def show_game_over_popup(screen):
    popup_width, popup_height = 300, 200
    popup_x = (WINDOW_WIDTH - popup_width) // 2
    popup_y = (WINDOW_HEIGHT - popup_height) // 2
    
    popup_surface = pygame.Surface((popup_width, popup_height))
    popup_surface.fill(WHITE)
    pygame.draw.rect(popup_surface, BLACK, (0, 0, popup_width, popup_height), 2)

    font = pygame.font.Font(None, 36)
    text = font.render("Game Over!!!", True, BLACK)
    text_rect = text.get_rect(center=(popup_width // 2, 50))
    popup_surface.blit(text, text_rect)

    button_width, button_height = 120, 40
    restart_button = pygame.Rect((popup_width - 2 * button_width - 20) // 2, 100, button_width, button_height)
    quit_button = pygame.Rect(restart_button.right + 20, 100, button_width, button_height)

    pygame.draw.rect(popup_surface, GREEN, restart_button)
    pygame.draw.rect(popup_surface, RED, quit_button)

    restart_text = font.render("Restart", True, BLACK)
    quit_text = font.render("Quit", True, BLACK)
    popup_surface.blit(restart_text, restart_text.get_rect(center=restart_button.center))
    popup_surface.blit(quit_text, quit_text.get_rect(center=quit_button.center))

    screen.blit(popup_surface, (popup_x, popup_y))
    pygame.display.flip()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                adjusted_pos = (mouse_pos[0] - popup_x, mouse_pos[1] - popup_y)
                if restart_button.collidepoint(adjusted_pos):
                    return "restart"
                elif quit_button.collidepoint(adjusted_pos):
                    return "quit"
        
        pygame.display.flip()  # 确保弹窗持续显示

# 初始化DQN代理
agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

def create_env(difficulty=0):
    """根据难度创建环境"""
    try:
        # 创建起始平台和终点平台
        spawn_platform = Platform(0, WINDOW_HEIGHT - 150, 100, 150)
        end_platform = Platform(WINDOW_WIDTH - 100, WINDOW_HEIGHT - 450, 100, 20)
        player = Player(spawn_platform)
        
        # 基础平台列表（包含起始和终点平台）
        platforms = [spawn_platform, end_platform]
        
        # 根据难度添加不同的拓展平台
        if difficulty == 0:
            # 难度0的拓展平台
            platforms.extend([
                Platform(100, 500, 400, 10),
                Platform(550, 425, 400, 10),
                Platform(1000, 350, 400, 10)
            ])
        elif difficulty == 1:
            # 难度1的拓展平台
            platforms.extend([
                Platform(150, 500, 200, 10),
                Platform(400, 465, 200, 10),
                Platform(650, 425, 200, 10),
                Platform(900, 385, 200, 10),
                Platform(1150, 350, 200, 10)
            ])
        elif difficulty == 2:
            # 难度2的拓展平台
            platforms.extend([
                Platform(100, 500, 100, 10),
                Platform(250, WINDOW_HEIGHT - 300, 100, 10),
                Platform(350, WINDOW_HEIGHT - 150, 150, 10),
                Platform(470, WINDOW_HEIGHT - 275, 150, 10),
                Platform(575, WINDOW_HEIGHT - 200, 150, 10),
                Platform(650, 350, 150, 10),
                Platform(800, WINDOW_HEIGHT - 250, 150, 10),
                Platform(925, 400, 100, 10),
                Platform(950, 300, 150, 10),
                Platform(1075, WINDOW_HEIGHT - 200, 100, 10),
                Platform(1250, WINDOW_HEIGHT - 250, 200, 10),
                Platform(1150, WINDOW_HEIGHT - 350, 150, 10)
            ])
        
        # 创建并返回游戏环境实例
        return GameEnvironment(player, platforms, [], end_platform)
    except Exception as e:
        print(f"Error creating environment with difficulty {difficulty}: {e}")
        # 如果创建特定难度失败，回退到难度0
        if difficulty > 0:
            print(f"Falling back to difficulty 0")
            return create_env(0)
        else:
            raise  # 如果难度0也失败，则抛出异

def evaluate_model(agent, difficulty=2, num_episodes=10):
    """评估模型性能"""
    global train

    total_rewards = 0
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # 禁用探索
    original_train_status = train
    train = False  # 禁用训练模式

    for _ in range(num_episodes):
        env = create_env(difficulty)
        episode_reward = 0
        state = env.get_state()
        done = False
        
        while not done:
            # 使用当前策略选择动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.model(state_tensor).argmax().item()
            
            # 执行动作
            if action == 0:
                env.player.move("left", env.platforms)
            elif action == 1:
                env.player.move("right", env.platforms)
            elif action == 2:
                env.player.jump()
            
            # 更新环境状态
            env.player.update(env.platforms)
            
            # 获取奖励和下一个状态
            reward = env.calculate_reward()
            next_state = env.get_state()
            done = env.player.has_fallen() or env._check_win()
            
            # 累计奖励
            episode_reward += reward
            state = next_state

        total_rewards += episode_reward

    # 恢复原始设置
    agent.epsilon = original_epsilon
    train = original_train_status
    return total_rewards / num_episodes

def start_game(episode=0, difficulty=0):
    clock = pygame.time.Clock()
    start_time = pygame.time.get_ticks()
    
    global train

    # 创建游戏环境
    game_env = create_env(difficulty)

    # 时间计数
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    frame_count = 0
    
    # 用于显示的变量
    current_action = None
    current_reward = 0
    current_state_display = {}
    
    # 动作映射
    action_names = {0: "Left", 1: "Right", 2: "Jump"}

    running = True
    while running:
        frame_count += 1
        dt = clock.tick(60 if not train else 200)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return "quit", 0

        current_time = pygame.time.get_ticks()
        
        # 更新子弹 (每3帧更新一次，减少计算负担)
        if frame_count % 3 == 0:
            game_env.update_bullets()
            
        if game_env.check_bullet_collisions():
            return "game_over", (current_time - start_time) / 1000

        # 获取当前状态
        current_state = game_env.get_state()
        
        # 更新状态显示
        current_state_display = {
            "Position X": f"{game_env.player.x:.1f}",
            "Position Y": f"{game_env.player.y:.1f}",
            "Velocity X": f"{game_env.player.vel.x:.1f}",
            "Velocity Y": f"{game_env.player.vel.y:.1f}",
            "On Platform": "Yes" if game_env.player.on_platform else "No",
            "Next Platform": f"{game_env._get_next_platform_distance():.1f}",
            "Next Platform Height": f"{game_env._get_next_platform_height():.1f}",
            "Goal X Distance": f"{game_env.end_platform.x - game_env.player.x:.1f}",
            "Goal Y Distance": f"{game_env.end_platform.y - game_env.player.y:.1f}"
        }

        # 使用DQN选择动作
        if train:
            action = agent.act(current_state)  # 使用agent的act方法，包含epsilon-greedy
        else:
            # 测试模式：完全使用神经网络决策
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_values = agent.model(state_tensor).cpu().data.numpy()[0]
                action = np.argmax(q_values)
                
                # 保存所有动作的Q值用于显示
                q_values_display = {action_names[i]: f"{q:.2f}" for i, q in enumerate(q_values)}
                
        current_action = action_names[action]

        # 执行动作
        if action == 0:
            game_env.player.move("left", game_env.platforms)
        elif action == 1:
            game_env.player.move("right", game_env.platforms)
        elif action == 2:
            game_env.player.jump()

        # 更新游戏状态
        game_env.player.update(game_env.platforms)
        
        # 计算奖励
        reward = game_env.calculate_reward()
        current_reward = reward
        
        # 获取新状态
        next_state = game_env.get_state()
        
        # 检查是否结束
        done = game_env.player.has_fallen() or game_env._check_win()
        
        # 存储经验
        if train:
            agent.remember(current_state, action, reward, next_state, done)
            
            # 训练网络 (每4帧训练一次，减少计算负担)
            if len(agent.memory) > agent.batch_size and frame_count % 4 == 0:
                agent.replay(agent.batch_size)

        # 渲染逻辑
        if not train:
            screen.fill(WHITE)
            for platform in game_env.platforms:
                platform.draw(screen)
            for bullet in game_env.bullets:
                bullet.draw(screen)
            game_env.player.draw(screen)
            
            # 显示时间
            elapsed_time = (pygame.time.get_ticks() - start_time) / 1000
            time_text = font.render(f"Time: {elapsed_time:.2f}", True, BLACK)
            screen.blit(time_text, (WINDOW_WIDTH - 150, 10))
            
            # 显示难度
            difficulty_text = font.render(f"Difficulty: {difficulty}", True, BLACK)
            screen.blit(difficulty_text, (WINDOW_WIDTH - 150, 50))
            
            # 显示当前动作
            action_text = font.render(f"Action: {current_action}", True, BLUE)
            screen.blit(action_text, (20, 20))
            
            # 显示当前奖励
            reward_color = GREEN if current_reward > 0 else RED
            reward_text = font.render(f"Reward: {current_reward:.2f}", True, reward_color)
            screen.blit(reward_text, (20, 60))
            
            # 显示状态信息
            y_offset = 100
            for key, value in current_state_display.items():
                state_text = small_font.render(f"{key}: {value}", True, BLACK)
                screen.blit(state_text, (20, y_offset))
                y_offset += 25
                
            # 显示Q值
            y_offset = 100
            q_text = small_font.render("Q Values:", True, BLACK)
            screen.blit(q_text, (200, y_offset))
            y_offset += 25
            
            for action_name, q_value in q_values_display.items():
                color = BLUE if action_name == current_action else BLACK
                q_value_text = small_font.render(f"{action_name}: {q_value}", True, color)
                screen.blit(q_value_text, (200, y_offset))
                y_offset += 25
            
            pygame.display.flip()
        
        if done:
            break

    total_time = (pygame.time.get_ticks() - start_time) / 1000
    return "win" if game_env._check_win() else "lose", round(total_time, 2)

def main():
    pygame.init()
    global screen, train
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("DQN Platform Game")
    
    if train:
        # 训练阶段
        num_episodes = 1000
        best_avg_reward = -np.inf
        difficulty = 0

        for episode in range(num_episodes):
            # 难度切换逻辑优化
            if episode > 0 and episode % 200 == 0 and difficulty < 2:
                difficulty += 1  # 每200轮增加难度
                print(f"Difficulty increased to {difficulty}")

                # 强制垃圾回收
                import gc
                gc.collect()

                # 重置环境和相关变量
                agent.memory = deque(maxlen=50000)  # 可选：重置记忆
                
                # 保存中间模型
                torch.save(agent.model.state_dict(), f'model_before_difficulty_{difficulty}.pth')

            print(f"Episode {episode + 1}, Difficulty: {difficulty}")
            
            try:
                result, time_spent = start_game(episode=episode, difficulty=difficulty)
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                continue
            
            # 每10轮更新目标网络
            if episode % 10 == 0:
                agent.update_target_model()
                
            # 保存模型
            if episode % 50 == 0:
                model_file = get_train_model_file(episode)
                torch.save(agent.model.state_dict(), model_file)
                print(f"Model saved as {model_file}")

            # 每100轮评估一次
            if episode % 100 == 0:
                test_reward = evaluate_model(agent)  # 需要实现评估函数
                if test_reward > best_avg_reward:
                    torch.save(agent.model.state_dict(), 'best_model.pth')
                    best_avg_reward = test_reward

            # 动态调整学习率
            if episode > 500:
                for g in agent.optimizer.param_groups:
                    g['lr'] = 0.0001

            # 打印训练进度
            print(f"Episode {episode+1}, Epsilon: {agent.epsilon:.3f}, Time: {time_spent:.2f}s")

            agent.decay_epsilon()

    # 测试阶段
    else:
        best_model_path = test_model_file
        agent.model.load_state_dict(torch.load(best_model_path))  # 加载训练好的模型
        agent.target_model.load_state_dict(agent.model.state_dict())  # 同步目标网络
        agent.model.eval()  # 关闭dropout和batchnorm的随机性
        agent.epsilon = 0.0  # 完全禁用随机探索
        
        # 设置难度等级 (0: 简单, 1: 中等, 2: 困难)
        test_difficulty = 2  # 在这里修改难度参数
        
        while True:
            result, time_spent = start_game(difficulty=test_difficulty)  # 使用设定的难度
            if result == "quit":
                break

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
