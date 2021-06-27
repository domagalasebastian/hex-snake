import curses
import random
import argparse
from enum import Enum
from network import DQNetwork
from agents import Bot, Snake
from utils import Grid, Cell, KeyboardListener


class Game:
    class Mode(Enum):
        SINGLE = 0
        BOT = 1

    def __init__(self, rows: int = 11, cols: int = 29,
                 game_mode: Mode = Mode.SINGLE):
        """
        @param rows: number of rows in playing area
        @param cols: number of cols in playing area
        @param game_mode : Single or against Bot
        """
        self.game_mode = game_mode
        self.grid = Grid(rows, cols)
        self.players = [Snake(self.grid[rows - 2, 1],
                              Snake.Direction.NORTH_EAST)]

        if game_mode == self.Mode.BOT:
            bot = Bot(self.grid[self.grid.rows - 2, self.grid.cols - 2],
                      Bot.Direction.NORTH_WEST)
            self.players.append(bot)
            self.model = DQNetwork(gamma=0.99, n_actions=6, epsilon=0.01,
                                   batch_size=32, input_dims=15, train=False)

    def _insert_fruit(self):
        """
        Method to place new fruit when previous one was eaten.
        """
        cell = random.choice(self.grid[Cell.Type.NORMAL])
        cell.cell_type = Cell.Type.FRUIT

    def get_active_players(self):
        """
        Get list of players that have not lost yet.
        @return: list of players that have not lost yet
        """
        return list(filter(lambda x: not x.lost, self.players))

    def _play(self, stdscr):
        """
        Game implementation.
        @param stdscr: curses window object representing the entire screen
        """
        # Screen preparation
        stdscr.clear()
        stdscr.refresh()
        _, window_width = stdscr.getmaxyx()

        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_YELLOW, curses.COLOR_BLACK)

        self._insert_fruit()
        with KeyboardListener():
            while active_players := self.get_active_players():
                # Update postion for each player
                for player in active_players:
                    if isinstance(player, Bot):
                        observation, _ = player.get_observation(self.grid)
                        action = self.model.choose_action(observation)
                        player.update_direction(action)
                    else:
                        player.update_direction(KeyboardListener.key())

                    player.move(self)

                # Update score for each player
                for i, player in enumerate(self.players):
                    info = f"{player._snake_type.name} SCORE: {player.score}"
                    color = curses.color_pair(player._snake_type.value + 1)
                    stdscr.addstr(0, i * (window_width // 2), info, color)

                self.grid.display(stdscr)
                stdscr.refresh()
                curses.napms(400)

    def play(self):
        curses.wrapper(self._play)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bot",
                        help="Game against Bot", action="store_true")
    args = parser.parse_args()
    Game(game_mode=Game.Mode.BOT if args.bot else Game.Mode.SINGLE).play()
