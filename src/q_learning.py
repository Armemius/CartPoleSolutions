import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd


class QTable:
    def __init__(self):
        self.lower_bounds = [-2.4, -2.0, -0.42, -3.5]
        self.upper_bounds = [2.4, 2.0, 0.42, 3.5]
        self.buckets = [50, 50, 50, 50]
        self.table = {}

        self.observation_space = 4
        self.action_space = 2

    def discretize(self, state):
        discretized = []
        for i in range(len(state)):
            scaling = self.buckets[i] / (
                self.upper_bounds[i] - self.lower_bounds[i]
            )
            discretized.append(
                int((state[i] - self.lower_bounds[i]) * scaling)
            )
        return tuple(discretized)

    def update(self, state, action, reward, next_state, alpha, gamma):
        state = self.discretize(state)
        next_state = self.discretize(next_state)

        if state not in self.table:
            self.table[state] = np.zeros(self.action_space)

        if next_state not in self.table:
            self.table[next_state] = np.zeros(self.action_space)

        self.table[state][action] += alpha * (
            reward
            + gamma * np.max(self.table[next_state])
            - self.table[state][action]
        )

    def get_action(self, state):
        state = self.discretize(state)
        if state not in self.table:
            self.table[state] = np.zeros(self.action_space)

        return np.argmax(self.table[state])

    def dump_csv(self):
        data = []
        for state, actions in self.table.items():
            continuous_state = [
                self.lower_bounds[i]
                + (state[i] + 0.5)
                * (self.upper_bounds[i] - self.lower_bounds[i])
                / self.buckets[i]
                for i in range(len(state))
            ]
            row = continuous_state + list(actions)
            data.append(row)

        data.sort(key=lambda x: x[:4])
        columns = [
            "position",
            "velocity",
            "angle",
            "angular_velocity",
            "action_0",
            "action_1",
        ]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv("assets/q_table.csv", index=False, sep=";")


class QLearning:
    def __init__(self, alpha=0.2, gamma=0.9, epsilon=0.1):
        self.env = gym.make("CartPole-v1")
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = QTable()

        self.observation_space = 4
        self.action_space = 2

        self.episodesHistory = []
        self.scoresHistory = []

    def train(self, episodes=1000):
        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            score = 0
            while not done:
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(self.action_space)
                else:
                    action = self.q_table.get_action(state)

                next_state, reward, done, truncated, _ = self.env.step(action)
                score += reward
                self.q_table.update(
                    state,
                    action,
                    reward,
                    next_state,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
                state = next_state
                if truncated:
                    done = True

            self.episodesHistory.append(episode)
            self.scoresHistory.append(score)

            if episode % 100 == 0 and episode > 0:
                print(f"Episode: {episode}, Score: {score}")

    def test(self):
        env = gym.make("CartPole-v1", render_mode="human")
        state, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = self.q_table.get_action(state)
            state, reward, done, _, _ = env.step(action)
            score += reward
            env.render()
        print(f"Score: {score}")

    def plot(self):

        window = 500
        mean_scores = (
            pd.Series(self.scoresHistory)
            .rolling(window=window, center=True)
            .mean()
        )
        plt.plot(self.episodesHistory, mean_scores, "-")
        plt.title("Mean Scores over 500 episodes")

        coefs = np.polyfit(self.episodesHistory, self.scoresHistory, 1)
        trendline = np.poly1d(coefs)
        plt.plot(self.episodesHistory, trendline(self.episodesHistory), "r--")
        plt.title("Mean Scores over 500 episodes")
        plt.xlabel("Episodes")
        plt.ylabel("Scores")
        plt.savefig("assets/plot.png")
