import sys
import tty
import curses
import termios
import threading
from enum import Enum


class KeyboardListener:
    __key = "e"
    VALID_KEYS = ("w", "e", "q", "s", "a", "d")

    def __init__(self):
        self.config = termios.tcgetattr(sys.stdin)
        self.stop_thread = False

    @classmethod
    def key(cls):
        """
        Get last pressed key.
        @return: last pressed key
        """
        return cls.__key

    @classmethod
    def __set_key(cls, value):
        """
        Update key value.
        """
        cls.__key = value

    def __get_key(self):
        """
        Update key value based on keyboard input.
        """
        while not self.stop_thread:
            key = sys.stdin.read(1)[0]
            if key in self.VALID_KEYS:
                self.__set_key(key)

    def __enter__(self):
        tty.setcbreak(sys.stdin)
        self.thread = threading.Thread(target=self.__get_key)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_thread = True
        self.thread.join()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.config)


class Cell:
    class Type(Enum):
        NORMAL = 0
        BOUND = 1
        SNAKE = 2
        BOT = 3
        FRUIT = 4

    NAMES = (" ", "BOUND", "SNAKE", "BOT", "FRUIT")

    def __init__(self, x: int, y: int, cell_type: Type):
        """
        @param x: cell x coordinate
        @param y: cell y coordinate
        @param cell_type: cell type
        """
        self.x = x
        self.y = y
        self.__cell_type = cell_type
        self.__name = self.NAMES[cell_type.value]

    @property
    def name(self):
        """
        Get cell name.
        @return: cell name
        """
        return self.__name

    @property
    def cell_type(self):
        """
        Get cell type.
        @return: cell type
        """
        return self.__cell_type

    @cell_type.setter
    def cell_type(self, value: Type):
        """
        Set cell type and cell name.
        @param value: new cell type
        """
        self.__cell_type = value
        self.__name = self.NAMES[self.__cell_type.value]


class Grid(list):
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        super().__init__([Cell(i, j, Cell.Type.BOUND if self.__is_edge(i, j)
                               else Cell.Type.NORMAL)
                          for i in range(self.rows) for j in range(self.cols)])

    def __is_edge(self, i: int, j: int) -> bool:
        """
        Method to check if field is on edge.
        @param i: x axis coordinate
        @param j: y axis coordinate
        @return: True if cell is on edge, otherwise False
        """
        return not all((i, j, self.rows - 1 - i, self.cols - 1 - j))

    def display(self, stdscr):
        """
        Method to display grid (playing area).
        @param stdscr: curses window object representing the entire screen
        """
        cell = self[0, 0]
        color = curses.color_pair(cell.cell_type.value + 1)
        stdscr.addstr(1, 0, (" " * 9 + "_" * 5) * (self.cols // 2), color)
        for y in range(self.rows * 4 + 2):
            first_two_lines = y < 2
            line_beginning_character = " " if first_two_lines else "\\"

            x_offset = 0
            line = (line_beginning_character,
                    " " + line_beginning_character, " ", "")[y % 4]
            color = curses.color_pair(cell.cell_type.value + 1)
            stdscr.addstr(y + 2, x_offset, line, color)
            x_offset += len(line)

            for x in range(-int(self.cols / 2), int(self.cols / 2) + 1):
                ix = y // 4
                iy = x + abs(-int(self.cols / 2))
                if ix != self.rows and iy != self.cols:
                    cell = self[ix, iy]

                line = ("/" + cell.name.center(7) + "\\", " " * 7, "_" * 5,
                        "/" + 5 * " " + "\\")[(y + x * 2 + self.cols) % 4]
                color = curses.color_pair(cell.cell_type.value + 1)
                stdscr.addstr(y + 2, x_offset, line, color)
                x_offset += len(line)

            line = "//  "[3 if first_two_lines else y % 4]
            color = curses.color_pair(cell.cell_type.value + 1)
            stdscr.addstr(y + 2, x_offset, line, color)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return next(filter(lambda cell: (cell.x, cell.y) == key, self))
        elif isinstance(key, Cell.Type):
            return list(filter(lambda cell: cell.cell_type == key, self))
