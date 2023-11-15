from algorithm import Minimax
from Board import Board
from LoggerClass import Logger
import random


class Solver:
    """
    Solver class for playing tic-tac-toe in 19x19
    """
    def __init__(self, current_field: str):

        # ВОТ ТУТ Я НЕ ПОНИМАЮ ЧТО ЭТО ЗА ПЕРЕМЕННАЯ И ЧТО В НЕЕ ПЕРЕДАЕТСЯ ЧИНИ
        self.algo = Minimax(Board())
        # ВОТ ТУТ Я НЕ ПОНИМАЮ ЧТО ЭТО ЗА ПЕРЕМЕННАЯ И ЧТО В НЕЕ ПЕРЕДАЕТСЯ ЧИНИ

        self.logger = Logger('Solver')
        self.current_field = current_field

    def make_turn(self) -> str:
        """
        Calculate next move for tic-tac-toe AI
        :return: Field after the turn
        """
        matrix = Board.str_to_matrix(self.current_field)
        self.algo.board.matrix = matrix
        x, y = self.algo.calculate_next_move(depth=3)
        current_field = Board.matrix_to_str(matrix)
        field_after_turn = current_field[:x * 19 + y] + 'x' + current_field[x * 19 + y + 1:]
        return field_after_turn

    @staticmethod
    def make_random_move(current_field: str):
        """
        Make random move for tic-tac-toe
        :param current_field:
        :return: Field after the turn
        """
        ptr = random.randint(0, 360)
        while (current_field[ptr] != '_'):
            ptr = random.randint(0, 360)
        new_field = current_field[:ptr] + 'o' + current_field[ptr + 1:]
        return new_field

    def output(self, game_field: str):
        """
        Output current game field
        :param game_field:
        :return:
        """
        self.logger.send_message('Текущая доска', 'info')
        for i in range(19):
            for j in range(19):
                print(game_field[i * 19 + j], end='')
            print()
'''
test_board = Board()
starting_board = '_' * 361
solver = Solver(test_board, starting_board)
current_field = starting_board
for i in range(40):
    solver.logger.send_message(f'Ход {i+1}', 'info')
    current_field = Solver.make_random_move(current_field)
    current_field = Solver.make_turn(solver, current_field)
    solver.output(current_field)
'''



