from algorithm import Minimax
from Board import Board
import random

class Test:
    @staticmethod
    def make_algo_move(string_field):
        matrix = Board.str_to_matrix(string_field)
        x, y = Minimax.calculate_next_move(matrix, depth=3)

        string_field = Board.matrix_to_str(matrix)
        string_field = string_field[:x * 19 + y] + 'x' + string_field[x * 19 + y + 1:]
        return string_field

    @staticmethod
    def make_random_move(string_field):
        ptr = random.randint(0, 360)
        while (string_field[ptr] != '_'):
            ptr = random.randint(0, 360)
        string_field = string_field[:ptr] + 'o' + string_field[ptr + 1:]
        return string_field


    @staticmethod
    def output(string_field):
        print('current board_matrix:\n')
        for i in range(19):
            for j in range(19):
                print(string_field[i * 19 + j], end='')
            print()

if __name__ == '__main__':
    default_s = '_' * 361
    cur_s = default_s

    for i in range(40):
        print('\n turn ', i)
        cur_s = Test.make_random_move(cur_s)
        cur_s = Test.make_algo_move(cur_s)
        Test.output(cur_s)



