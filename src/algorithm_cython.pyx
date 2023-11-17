import time
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport rand
from cpython cimport array
#from Board import Board
cnp.import_array()

from libc.time cimport time, time_t
#from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t

#from libcpp cimport bool

cdef class Board:
    # black is an opponent, our bot goes white (no racism)
    @staticmethod
    cdef add_stone(cnp.ndarray matrix, int posX,int posY, int black):
        if black: 
            matrix[posY][posX] = 2
        else:
            matrix[posY][posX] = 1

    @staticmethod
    cdef remove_stone(cnp.ndarray[DTYPE_t, ndim = 2] matrix, int posX,int posY):
        matrix[posY][posX] = 0
        
    @staticmethod
    cdef clone_matrix(cnp.ndarray[DTYPE_t, ndim = 2] matrix):
        return cnp.ndarray.copy(matrix)

    @staticmethod
    #@numba.njit(parallel=True)
    def str_to_matrix(str str_field):

        cdef cnp.ndarray[DTYPE_t, ndim = 2] matrix = np.empty((19, 19), dtype=np.int64)
        cdef int idx = 0
        cdef str symb =''
        cdef str fig = str_field[361]
        cdef str op_fig = str_field[362]
        cdef int i = 0
        cdef int j = 0
        #print(fig)
        for i in range(19):
            j = 0
            for j in range(19):
                # i*19 + j
                idx = i*19+j
                symb = str_field[idx]
                if (symb==fig):
                        matrix[i][j] = 1
                elif (symb==op_fig):
                        matrix[i][j] = 2
        return matrix


    @staticmethod
    cdef list generate_moves(cnp.ndarray[DTYPE_t, ndim = 2] board_matrix):
        cdef list move_list = []
        cdef int board_size = len(board_matrix)

        # Look for cells that have at least one stone in an adjacent cell.
        cdef int[2] move = [0,0]
        cdef int i = 0
        cdef int j = 0
        for i in range(board_size):
            j = 0
            for j in range(board_size):
                if board_matrix[i][j] > 0:
                    continue

                if (i > 0):
                    if (j > 0):
                        if ((board_matrix[i - 1][j - 1] > 0 or board_matrix[i][j - 1] > 0)):# and board_matrix[i][j]==0):
                            move = [i, j]
                            move_list.append(move)
                            continue
                    if (j < board_size - 1):
                        if ((board_matrix[i - 1][j + 1] > 0 or board_matrix[i][j + 1] > 0)):# and board_matrix[i][j]==0):
                            move = [i, j]
                            move_list.append(move)
                            continue
                    if ((board_matrix[i - 1][j] > 0)):# and board_matrix[i][j]==0):
                        move = [i, j]
                        move_list.append(move)
                        continue

                if (i < board_size - 1):
                    if (j > 0):
                        if ((board_matrix[i + 1][j - 1] > 0 or board_matrix[i][j - 1] > 0)):# and board_matrix[i][j]==0):
                            move = [i, j]
                            move_list.append(move)
                            continue
                    if (j < board_size - 1):
                        if ((board_matrix[i + 1][j + 1] > 0 or board_matrix[i][j + 1] > 0)):# and board_matrix[i][j]==0):
                            move = [i, j]
                            move_list.append(move)
                            continue
                    if ((board_matrix[i + 1][j] > 0)):# and board_matrix[i][j]==0):
                        move = [i, j]
                        move_list.append(move)
                        continue

        return move_list


cdef class Minimax:
    #cdef int WIN_SCORE = 100_000_000
    @staticmethod
    cdef int get_win_score():
        return 100_000_000


    @staticmethod
    cdef float evaluate_board_for_white(cnp.ndarray[DTYPE_t, ndim = 2] board_matrix, int blackTurn):
        #self.evaluationCount+=1

        cdef float blackScore = Minimax.get_score(board_matrix, 1, blackTurn)
        cdef float whiteScore = Minimax.get_score(board_matrix, 0, blackTurn)

        if (blackScore==0):
            blackScore = 1.0

        return whiteScore/blackScore


    @staticmethod
    cdef float get_score(cnp.ndarray[DTYPE_t, ndim = 2] boardMatrix, int forBlack, int blacksTurn):
        # Read the board_matrix

        # Calculate score for each of the 3 directions
        return (Minimax.evaluate_horizontal(boardMatrix, forBlack, blacksTurn) +
                Minimax.evaluate_vertical(boardMatrix, forBlack, blacksTurn) +
                Minimax.evaluate_diagonal(boardMatrix, forBlack, blacksTurn))



    # This function is used to get the next intelligent move to make for the AI.
    #@numba.njit(parallel=True)
    @staticmethod
    def calculate_next_move(cnp.ndarray[DTYPE_t, ndim = 2] matrix, int depth):
        # Block the board_matrix for AI to make a decision.
        cdef int[2] move = [0, 0]

        # Used for benchmarking purposes only.
        '''
        cdef timespec ts
        cdef double start_time, current_time
        clock_gettime(CLOCK_REALTIME, &ts)
        start_time = ts.tv_sec + (ts.tv_nsec / 1000000000.)
        '''
        #cdef time_t start_time = time(NULL)
        # Check if any available move can finish the game to make sure the AI always
        # takes the opportunity to finish the game.
        cdef cnp.ndarray[DTYPE_t, ndim = 1] bestMove = np.array(Minimax.search_winning_move(matrix), dtype=np.int64)
        cdef cnp.ndarray[DTYPE_t, ndim = 2] tmp_board_matrix
        #print(bestMove, move)
        if (len(bestMove) != 0):
            # Finishing move is found.
            move[0] = bestMove[1]
            move[1] = bestMove[2]
        else:
            tmp_board_matrix = cnp.ndarray.copy(matrix) #Board.clone_matrix(matrix)

            # If there is no such move, search the minimax tree with specified depth.
            bestMove = np.array(Minimax.minimax_search_ab(depth, tmp_board_matrix, 1, -1.0, Minimax.get_win_score()), dtype=np.int64)
            #print('Best moves: ')
            #print(bestMove)
            move[0] = bestMove[1]
            move[1] = bestMove[2]
        #clock_gettime(CLOCK_REALTIME, &ts)
        #cdef time_t current_time = time(NULL)
        # print("Cases calculated: " + str(self.evaluationCount) + "
        #print("Calculation time: " + str(
        #    int((current_time - start_time) * 1000)) + " ms")
        return move

    """
     alpha: Best AI Move (Max)
     beta: Best Player Move (Min)
     returns: [score, move[0], move[1]]
    """
    @staticmethod
    #@numba.njit(parallel=True)
    #cdef minimax_search_ab(int depth, vector [vector[int]] dummy_board_matrix, bool max_player, int alpha, int beta):
    cdef list minimax_search_ab(int depth, cnp.ndarray[DTYPE_t, ndim = 2] dummy_board_matrix, int max_player,double alpha, double beta):
        # Last depth (terminal node), evaluate the current board_matrix score.
        cdef list ans = []
        if (depth == 0):
            #cdef int[4] ans = {Minimax.evaluate_board_for_white(dummy_board_matrix, not max_player), None, None}
            #return ans
            ans = [Minimax.evaluate_board_for_white(dummy_board_matrix, not max_player), 0, 0]
            return ans


        # Generate all possible moves from this node of the Minimax Tree
        cdef list  all_possible_moves = (Board.generate_moves(dummy_board_matrix))
        #print("all posible moves:")
        #print(all_possible_moves)
        cdef list tmp_arr = []
        # If there are no possible moves left, treat this node as a terminal node and return the score.
        if (len(all_possible_moves) == 0):
            tmp_arr = [Minimax.evaluate_board_for_white(dummy_board_matrix, not max_player), 0, 0]

            return tmp_arr
        
        cdef int rand_x = rand()%19, rand_y = rand()%19
        while (dummy_board_matrix[rand_y][rand_x]!=0):
             rand_x = rand()%19
             rand_y = rand()%19
        
        cdef int[3] best_move = [0, rand_y, rand_x]
        cdef int[3] temp_move = [0, 0, 0]
        # Generate Minimax Tree and calculate node scores.
        if (max_player):
            # Initialize the starting best move with -infinity.
            best_move[0] = -1

            # Iterate for all possible moves that can be made.
            for move in all_possible_moves:

                # Play the move on the temporary board_matrix without drawing anything.
                Board.add_stone(dummy_board_matrix, move[1], move[0], 0)

                # Call the minimax function for the next depth, to look for a minimum score.

                temp_move = Minimax.minimax_search_ab(depth - 1, dummy_board_matrix, 0, alpha, beta)

                # Backtrack and remove.
                Board.remove_stone(dummy_board_matrix, move[1], move[0])

                # Updating alpha (alpha value holds the maximum score)

                if (temp_move[0] > alpha):
                    alpha = temp_move[0]

                if (temp_move[0] >= beta):
                    return temp_move

                # Find the move with the maximum score.
                if (temp_move[0] > best_move[0]):
                    best_move = temp_move
                    best_move[1] = move[0]
                    best_move[2] = move[1]

        else:
            # Initialize the starting best move using the first move in the list and +infinity score.
            best_move[0] = 100_000_000
            best_move[1] = all_possible_moves[0][0]
            best_move[2] = all_possible_moves[0][1]

            # Iterate for all possible moves that can be made.
            for move in all_possible_moves:

                # Play the move on the temporary board_matrix without drawing anything.
                Board.add_stone(dummy_board_matrix, move[1], move[0], 1)

                temp_move = Minimax.minimax_search_ab(depth - 1, dummy_board_matrix, 1, alpha, beta)

                Board.remove_stone(dummy_board_matrix, move[1], move[0])

                beta = min(temp_move[0], beta)

                if temp_move[0] <= alpha:
                    return temp_move

                # Find the move with the minimum score.
                if temp_move[0] < best_move[0]:
                    best_move = temp_move
                    best_move[1] = move[0]
                    best_move[2] = move[1]
        # // Return the best move found in this depth
        return best_move


    @staticmethod
    cdef list search_winning_move(cnp.ndarray[DTYPE_t, ndim = 2] board_matrix):

        cdef list  all_possible_moves = Board.generate_moves(board_matrix)
        cdef int[3] winning_move = [0, 0, 0]

        cdef cnp.ndarray[DTYPE_t, ndim = 2] dummy_board
        #print(len(all_possible_moves))
        for move in all_possible_moves:
            # Create a temporary board_matrix that is equivalent to the current board_matrix
            dummy_board = Board.clone_matrix(board_matrix)
            # Play the move on that temporary board_matrix without drawing anything
            Board.add_stone(dummy_board, move[1], move[0], 0)

            # If the white player has a winning score in that temporary board_matrix, return the move.
            if (Minimax.get_score(dummy_board, 0, 0) >= Minimax.get_win_score()):
                winning_move[1] = move[0]
                winning_move[2] = move[1]
                #print("winning move: ", winning_move)
                return winning_move
        #print("search found nothin")
        return []

    # This function calculates the score by evaluating the stone positions in horizontal direction
    @staticmethod
    cdef int evaluate_horizontal(cnp.ndarray[DTYPE_t, ndim = 2] boardMatrix, int forBlack, int playersTurn):

        cdef int[3] evaluations = [0, 2, 0] # [0] -> consecutive count, [1] -> block count, [2] -> score
        
        cdef int i = 0
        cdef int j = 0
        for i in range(len(boardMatrix)):
            #// Iterate over all cells in a row
            j = 0
            for j in range( len(boardMatrix[0])):
                #// Check if the selected player has a stone in the current cell
                evaluations = Minimax.evaluate_directions(boardMatrix, i, j, forBlack, playersTurn, evaluations)

            evaluations = Minimax.evaluate_directions_after_one_pass(evaluations, forBlack, playersTurn)

        return evaluations[2]

    '''
    This function calculates the score by evaluating the stone positions in vertical direction
     The procedure is the exact same of the horizontal one.
    '''

    @staticmethod
    cdef int evaluate_vertical(cnp.ndarray[DTYPE_t, ndim = 2] boardMatrix, int forBlack,int playersTurn):
        cdef int[3] evaluations = [0, 2, 0] # [0] -> consecutive count, [1] -> block count, [2] -> score
        cdef int j = 0
        cdef int i = 0
        for j in range(len(boardMatrix[0])):
            i = 0
            for i in range(len(boardMatrix)):
                evaluations = Minimax.evaluate_directions(boardMatrix, i, j, forBlack, playersTurn, evaluations)
            evaluations = Minimax.evaluate_directions_after_one_pass(evaluations, forBlack, playersTurn)
        return evaluations[2]

    @staticmethod
    cdef int evaluate_diagonal(cnp.ndarray[DTYPE_t, ndim = 2] boardMatrix, int forBlack, int playersTurn):
        cdef int[3] evaluations = [0, 2, 0]  # [0] -> consecutive count, [1] -> block count, [2] -> score
        # From bottom-left to top-right diagonally
        cdef int iStart = 0
        cdef int iEnd = 0

        cdef int i = 0
        cdef int k = 0
        for k in range(0, 2 * (len(boardMatrix) - 1) + 1):
            iStart = max(0, k - len(boardMatrix) + 1)
            iEnd = min(len(boardMatrix) - 1, k)
            i = 0
            for i in range(iStart, iEnd + 1):
                evaluations = Minimax.evaluate_directions(boardMatrix, i, k - i, forBlack, playersTurn, evaluations)
            evaluations = Minimax.evaluate_directions_after_one_pass(evaluations, forBlack, playersTurn)

        # From top-left to bottom-right diagonally
        i = 0
        k = 0
        for k in range(1 - len(boardMatrix), len(boardMatrix)):
            iStart = max(0, k)
            iEnd = min(len(boardMatrix) + k - 1, len(boardMatrix) - 1)
            i = 0
            for i in range(iStart, iEnd + 1):
                evaluations = Minimax.evaluate_directions(boardMatrix, i, i - k, forBlack, playersTurn, evaluations)
            evaluations = Minimax.evaluate_directions_after_one_pass(evaluations, forBlack, playersTurn)

        return evaluations[2]

    @staticmethod
    cdef int*  evaluate_directions(cnp.ndarray[DTYPE_t, ndim = 2] boardMatrix, int i,int j, int isBot,int botsTurn, int*  eval):
        # Check if the selected player has a stone in the current cell
        if boardMatrix[i][j] == (2 if isBot else 1):
            # Increment consecutive stones count
            eval[0] += 1
        # Check if cell is empty
        elif boardMatrix[i][j] == 0:
            # Check if there were any consecutive stones before this empty cell
            if eval[0] > 0:
                # Consecutive set is not blocked by opponent, decrement block count
                eval[1] -= 1
                # Get consecutive set score
                eval[2] += Minimax.get_consecutive_set_score(eval[0], eval[1], isBot == botsTurn)
                # Reset consecutive stone count
                eval[0] = 0
                # Current cell is empty, next consecutive set will have at most 1 blocked side.
            # No consecutive stones.
            # Current cell is empty, next consecutive set will have at most 1 blocked side.
            eval[1] = 1
        # Cell is occupied by opponent
        # Check if there were any consecutive stones before this empty cell
        elif eval[0] > 0:
            # Get consecutive set score
            eval[2] += Minimax.get_consecutive_set_score(eval[0], eval[1], isBot == botsTurn)
            # Reset consecutive stone count
            eval[0] = 0
            # Current cell is occupied by opponent, next consecutive set may have 2 blocked sides
            eval[1] = 2
        else:
            # Current cell is occupied by opponent, next consecutive set may have 2 blocked sides
            eval[1] = 2
        return eval

    @staticmethod
    cdef int*  evaluate_directions_after_one_pass(int[3] eval,int isBot, int playersTurn):
        # End of row, check if there were any consecutive stones before we reached right border
        if eval[0] > 0:
            eval[2] += Minimax.get_consecutive_set_score(eval[0], eval[1], isBot == playersTurn)
        # Reset consecutive stone and blocks count
        eval[0] = 0
        eval[1] = 2
        return eval


    # This function returns the score of a given consecutive stone set.
    # count: Number of consecutive stones in the set
    # blocks: Number of blocked sides of the set (2: both sides blocked, 1: single side blocked, 0: both sides free)
    @staticmethod
    cdef int get_consecutive_set_score(int count, int blocks,int currentTurn):
        cdef int winGuarantee = 1000000

        # If both sides of a set are blocked, this set is worthless return 0 points.
        if (blocks == 2 and count < 5):
            return 0

        if (count == 5):
            # 5 consecutive wins the game
            return Minimax.get_win_score()
        elif (count == 4):
            # 4 consecutive stones in the user's turn guarantees a win.
            # (User can win the game by placing the 5th stone after the set)
            if (currentTurn):
                return winGuarantee
            else:
                # Opponent's turn
                # If neither side is blocked, 4 consecutive stones guarantees a win in the next turn.
                if (blocks == 0):
                    return winGuarantee // 4
                # If only a single side is blocked, 4 consecutive stones limits the opponents move
                # (Opponent can only place a stone that will block the remaining side, otherwise the game is lost
                # in the next turn). So a relatively high score is given for this set.
                else:
                    return 200
        elif (count) == 3:
            # 3 consecutive stones
            if blocks == 0:
                # Neither side is blocked.
                # If it's the current player's turn, a win is guaranteed in the next 2 turns.
                # (User places another stone to make the set 4 consecutive, opponent can only block one side)
                # However, the opponent may win the game in the next turn, therefore this score is lower than win
                # guaranteed scores but still a very high score.
                if (currentTurn):
                    return 50_000
                # If it's the opponent's turn, this set forces the opponent to block one of the sides of the set.
                # So a relatively high score is given for this set.
                else:
                    return 200
            else:
                # One of the sides is blocked.
                # Playmaker scores
                if (currentTurn):
                    return 10
                else:
                    return 5
        elif (count == 2):
            # 2 consecutive stones
            # Playmaker scores
            if (blocks == 0):
                if (currentTurn):
                    return 7
                else:
                    return 5
            else:
                return 3
        elif (count == 1):
            return 1

        # More than 5 consecutive stones?
        return Minimax.get_win_score() * 2



#if __name__=="main":
#    Minimax.calculate_next_move()