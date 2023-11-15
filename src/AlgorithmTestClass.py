from algorithm import Minimax, MinimaxCalculations
from Board import Board, BoardCalculations
import random

class Test:
    def __init__(self, board=None, s_field=None):
        self.cur_s = s_field
        self.board = Board(board)
        self.algo = Minimax(board)


    def make_algo_move(self, s_field):
        matrix = BoardCalculations.str_to_matrix(s_field)
        self.algo.board.matrix = matrix
        x, y = self.algo.calculate_next_move(depth=3)
        #matrix[y][x]=2
        s_field = BoardCalculations.matrix_to_str(matrix)
        s_field = s_field[:x*19+y] + 'x' + s_field[x*19+y+1:]
        return s_field

    @staticmethod
    def make_random_move(s_field):
        ptr = random.randint(0, 360)
        while (s_field[ptr] != '_'):
            ptr = random.randint(0, 360)
        s_field = s_field[:ptr] + 'o' + s_field[ptr + 1:]
        return s_field


    @staticmethod
    def output(s_field):
        print('current board:\n')
        for i in range(19):
            for j in range(19):
                print(s_field[i*19+j], end='')
            print()

if __name__ == '__main__':
    test_board = Board()
    default_s = '_' * 361
    test = Test(test_board, default_s)
    cur_s = default_s

    for i in range(40):
        print('\n turn ', i)
        cur_s = Test.make_random_move(cur_s)
        cur_s = Test.make_algo_move(test, cur_s)
        Test.output(cur_s)



