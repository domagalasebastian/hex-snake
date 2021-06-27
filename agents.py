from enum import Enum
from collections import deque
from utils import Cell, Grid, KeyboardListener


class Snake:
    class Direction(Enum):
        NORTH = 0
        NORTH_EAST = 1
        NORTH_WEST = 2
        SOUTH = 3
        SOUTH_WEST = 4
        SOUTH_EAST = 5

    KEY_TO_DIRECTION = dict(zip(KeyboardListener.VALID_KEYS, Direction))

    def __init__(self, head: Cell,
                 current_direction: Direction,
                 snake_type: Cell.Type = Cell.Type.SNAKE):
        """
        @param head: head field
        @param current_direction: snake starting direction
        @param snake_type: type of snake (player or bot)
        """
        self._current_direction = current_direction
        self._head = self._previous_head = head
        self._head.cell_type = self._snake_type = snake_type
        self._score = 0
        self._body = deque([self._head])
        self._lost = False

    @property
    def score(self):
        """
        Get player score.
        @return: player score.
        """
        return self._score

    @property
    def lost(self):
        """
        Get player state.
        @return: player state.
        """
        return self._lost

    def is_opposite_direction(self, direction_value: int):
        """
        Method to check if requested direction is opposite to current one.
        @param direction_value: requested direction
        @return: True if given direction is opposite to the current one,
                 otherwise False.
        """
        return self._current_direction.value in (direction_value - 3,
                                                 direction_value + 3)

    def update_direction(self, key: str):
        """
        Method to update direction based on last key pressed.
        @param key: last valid key pressed
        """
        new_direction = self.KEY_TO_DIRECTION[key]
        if not self.is_opposite_direction(new_direction.value):
            self._current_direction = new_direction

    def _head_position_change(self, direction: Direction):
        """
        Method to determine head shift based on current direction
        and head position.
        @param direction: requested direction
        @return: head shift
        """
        position_change = {
            self.Direction.NORTH_EAST: ((0, 1), (-1, 1)),
            self.Direction.NORTH_WEST: ((0, -1), (-1, -1)),
            self.Direction.SOUTH_EAST: ((1, 1), (0, 1)),
            self.Direction.SOUTH_WEST: ((1, -1), (0, -1)),
            self.Direction.NORTH: ((-1, 0), (-1, 0)),
            self.Direction.SOUTH: ((1, 0), (1, 0))
        }

        return position_change[direction][self._head.y % 2]

    def move(self, game):
        """
        Method to update snake position based on current direction.
        @param game: Game object
        """
        # Determine new head position
        x_delta, y_delta = self._head_position_change(self._current_direction)
        new_head = game.grid[self._head.x + x_delta, self._head.y + y_delta]

        tail = self._body[-1]
        for i in range(len(self._body) - 1, 0, -1):
            self._body[i] = self._body[i - 1]

        # Check collision and fruit eaten conditions
        if new_head.cell_type.value in range(1, 4):
            self._body.append(tail)
            return self.remove_body()
        elif new_head.cell_type == Cell.Type.FRUIT:
            self._score += 10
            self._body.append(tail)
            game._insert_fruit()
        else:
            tail.cell_type = Cell.Type.NORMAL

        # Update head field
        new_head.cell_type = self._snake_type
        self._previous_head = self._head
        self._head = self._body[0] = new_head

    def remove_body(self):
        """
        Method to remove dead snake from playing area.
        """
        for cell in self._body:
            cell.cell_type = Cell.Type.NORMAL

        self._body.clear()
        self._lost = True


class Bot(Snake):
    def __init__(self, head: Cell,
                 current_direction: Snake.Direction,
                 snake_type: Cell.Type = Cell.Type.BOT):
        """
        @param head: head field
        @param current_direction: snake starting direction
        @param snake_type: type of snake (player or bot)
        """
        super().__init__(head, current_direction, snake_type=snake_type)
        self.__previous_score = self._score

    def update_direction(self, action: int):
        """
        Method to update direction based on predicted action.
        @param key: predicted action
        """
        new_direction = self.Direction(action)
        if not self.is_opposite_direction(new_direction.value):
            self._current_direction = new_direction

    def get_observation(self, grid: Grid):
        """
        Method to get game state for predicting purposes.
        @param grid: Grid object
        @return: reward for action and post move observation
        """
        state = []
        # Check if fields around head are available
        for direction in self.Direction:
            x_change, y_change = self._head_position_change(direction)
            cell = grid[self._head.x + x_change, self._head.y + y_change]
            state.append(int(cell.cell_type.value in range(1, 4)))

        fruit_position = [0] * 8
        fruit = grid[Cell.Type.FRUIT][0]

        # Use one-hot encoding to define fruit direction
        if fruit.x > self._head.x:
            if fruit.y > self._head.y:
                fruit_position[3] = 1  # SE
            elif fruit.y < self._head.y:
                fruit_position[5] = 1  # SW
            elif fruit.y == self._head.y:
                fruit_position[4] = 1  # S
        elif fruit.x < self._head.x:
            if fruit.y > self._head.y:
                fruit_position[1] = 1  # NE
            elif fruit.y < self._head.y:
                fruit_position[7] = 1  # NW
            elif fruit.y == self._head.y:
                fruit_position[0] = 1  # N
        elif fruit.x == self._head.x:
            if fruit.y > self._head.y:
                fruit_position[2] = 1  # E
            elif fruit.y < self._head.y:
                fruit_position[6] = 1  # W

        state += fruit_position
        current_distance = self.__calculate_distance(fruit, self._head)
        if self._score > self.__previous_score:
            reward = 10
            self.__previous_score = self._score
        else:
            previous_distance = self.__calculate_distance(fruit,
                                                          self._previous_head)
            reward = 2 if current_distance < previous_distance else -3

        state.append(current_distance)

        return state, reward

    @staticmethod
    def __calculate_distance(fruit: Cell, head: Cell):
        """
        Method to calculate distance between cells.
        @param fruit: fruit field
        @param head: head field
        @return: distance between cells
        """
        return ((fruit.x - head.x) ** 2 + (fruit.y - head.y) ** 2)**0.5
