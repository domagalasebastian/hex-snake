import random
from utils import Cell
from snake import Game
from agents import Bot
from network import DQNetwork
from tensorflow.python.framework.ops import disable_eager_execution


if __name__ == '__main__':
    disable_eager_execution()
    model = DQNetwork(gamma=0.99, n_actions=6, epsilon=1.0,
                      batch_size=32, input_dims=15)
    episodes = 10000
    scores = []

    for i in range(episodes):
        score = counter = 0
        game = Game()
        bot = Bot(game.grid[game.grid.rows - 2, game.grid.cols - 2],
                  Bot.Direction.NORTH_WEST)

        # Add inactive bots as an obstacles
        for _ in range(10):
            Bot(random.choice(game.grid[Cell.Type.NORMAL]),
                Bot.Direction.NORTH_WEST)

        game._insert_fruit()
        bot.move(game)

        observation, _ = bot.get_observation(game.grid)
        while not bot.lost:
            action = model.choose_action(observation)
            bot.update_direction(action)
            bot.move(game)

            observation_, reward = (([0] * 15, -10) if bot.lost
                                    else bot.get_observation(game.grid))
            score += reward

            transition = (observation, action, reward, observation_, bot.lost)
            model.store_transition(transition)
            observation = observation_
            model.train()
            counter += 1
            if counter > 5000:
                break

        scores.append(score)

        avg_score = sum(scores[-100:]) / 100
        info = f"""Episode: {i}, score: {score},
                   avg score in last 100 episodes: {avg_score},
                   fruits: {bot.score / 10}"""
        print(info)

    model.save_model()
