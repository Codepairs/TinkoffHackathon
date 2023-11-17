#from algorithm import Minimax
from Cython.Build import cythonize

import time
import numpy as np
import pyximport
pyximport.install()
from algorithm_cython import Minimax, Board

#from Board import Board
import random



class Test:
    @staticmethod
    def make_algo_move(string_field, figure, opponent_figure):
        
        field_with_figure = string_field+figure+opponent_figure
        matrix = Board.str_to_matrix(field_with_figure)
        #print(len(field_with_figure))
        #print(len(matrix), len(matrix[0]))
        start_time = time.time()
        x, y = Minimax.calculate_next_move(matrix, depth=2)
        x = (int) (x)
        y = (int) (y)
        #print(x,y)
        end_time = time.time()
        #print("Calculation time: " + str(
        #    int((end_time - start_time) * 1000)) + " ms")

        string_field = string_field[:x * 19 + y] + figure + string_field[x * 19 + y + 1:]
        #print(len(string_field))
        
        return string_field

    @staticmethod
    def make_random_move(string_field, figure):
        ptr = random.randint(0, 360)
        while (string_field[ptr] != '_'):
            ptr = random.randint(0, 360)
        string_field = string_field[:ptr] + figure + string_field[ptr + 1:]
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
        #print(len(cur_s))
        cur_s = Test.make_random_move(cur_s, 'x')
        #print(len(cur_s))
        cur_s = Test.make_algo_move(cur_s, 'o', 'x')
        #print(len(cur_s))
        Test.output(cur_s)