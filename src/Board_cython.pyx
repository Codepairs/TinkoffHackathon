import copy
from Board import Board
import time
import numpy as np
cimport numpy as cnp

cnp.import_array()
DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t

class Board:
    # black is an opponent, our bot goes white (no racism)
    @staticmethod
    def add_stone(matrix, posX, posY, black: bool):
        matrix[posY][posX] = 2 if black else 1

    @staticmethod
    def remove_stone(matrix, posX, posY):
        matrix[posY][posX] = 0
        
    @staticmethod
    def clone_matrix(cnp.ndarray matrix):
        return matrix.copy()

    @staticmethod
    #@numba.njit(parallel=True)
    def str_to_matrix(str_field):
        matrix = np.zeros((19, 19))
        for i in range(19):
            for j in range(19):
                # i*19 + j
                idx = i*19+j
                match(str_field[idx]):
                    case 'x':
                        matrix[i][j] = 2
                    case 'o':
                        matrix[i][j] = 1
        return matrix

    @staticmethod
    #@numba.njit(parallel=True)
    def matrix_to_str(matrix):
        str_field = ''
        for i in range(19):
            for j in range(19):
                # i*19 + j
                idx = i * 19 + j
                match (matrix[i][j]):
                    case 2:
                        str_field += 'x'
                    case 1:
                        str_field += 'o'
                    case 0:
                        str_field += '_'
        return str_field

    @staticmethod
    def generate_moves(board_matrix):
        move_list = []
        board_size = len(board_matrix)

        # Look for cells that have at least one stone in an adjacent cell.
        for i in range(board_size):
            for j in range(board_size):
                if board_matrix[i][j] > 0:
                    continue

                if i > 0:
                    if j > 0:
                        if board_matrix[i - 1][j - 1] > 0 or board_matrix[i][j - 1] > 0:
                            move = [i, j]
                            move_list.append(move)
                            continue
                    if j < board_size - 1:
                        if board_matrix[i - 1][j + 1] > 0 or board_matrix[i][j + 1] > 0:
                            move = [i, j]
                            move_list.append(move)
                            continue
                    if board_matrix[i - 1][j] > 0:
                        move = [i, j]
                        move_list.append(move)
                        continue

                if i < board_size - 1:
                    if j > 0:
                        if board_matrix[i + 1][j - 1] > 0 or board_matrix[i][j - 1] > 0:
                            move = [i, j]
                            move_list.append(move)
                            continue
                    if j < board_size - 1:
                        if board_matrix[i + 1][j + 1] > 0 or board_matrix[i][j + 1] > 0:
                            move = [i, j]
                            move_list.append(move)
                            continue
                    if board_matrix[i + 1][j] > 0:
                        move = [i, j]
                        move_list.append(move)
                        continue

        return move_list

