from Board import Board
import time

class Minimax:
    WIN_SCORE = 100_000_000
    @staticmethod
    def get_win_score():
        return Minimax.WIN_SCORE

    '''
    // This function calculates the relative score of the white player against the black.
    // (i.e. how likely is white player to win the game before the black player)
    // This value will be used as the score in the Minimax algorithm.
    '''
    @staticmethod
    def evaluate_board_for_white(board, blackTurn: bool):
        #self.evaluationCount+=1

        blackScore = Minimax.get_score(board, True, blackTurn)
        whiteScore = Minimax.get_score(board, False, blackTurn)

        if (blackScore==0):
            blackScore = 1.0

        return whiteScore/blackScore


    '''
    // This function calculates the board_matrix score of the specified player.
    // (i.e. How good a player's general standing on the board_matrix by considering how many 
    //  consecutive 2's, 3's, 4's it has, how many of them are blocked etc...)
    '''
    @staticmethod
    def get_score(boardMatrix, forBlack: bool, blacksTurn: bool):
        # Read the board_matrix

        # Calculate score for each of the 3 directions
        return Minimax.evaluate_horizontal(boardMatrix, forBlack, blacksTurn) + \
                Minimax.evaluate_vertical(boardMatrix, forBlack, blacksTurn) + \
                Minimax.evaluate_diagonal(boardMatrix, forBlack, blacksTurn)



    # This function is used to get the next intelligent move to make for the AI.
    #@numba.njit(parallel=True)
    @staticmethod
    def calculate_next_move(matrix, depth):
        # Block the board_matrix for AI to make a decision.
        move = [0, 0]

        # Used for benchmarking purposes only.
        startTime = time.time()

        # Check if any available move can finish the game to make sure the AI always
        # takes the opportunity to finish the game.
        bestMove = Minimax.search_winning_move(matrix)

        if bestMove != None:
            # Finishing move is found.
            move[0] = bestMove[1]
            move[1] = bestMove[2]
        else:
            tmp_board_matrix = Board.clone_matrix(matrix)

            # If there is no such move, search the minimax tree with specified depth.
            bestMove = Minimax.minimax_search_ab(depth, tmp_board_matrix, True, -1.0, Minimax.get_win_score())
            if bestMove[1] == None:
                move = None
            else:
                move[0] = bestMove[1]
                move[1] = bestMove[2]

        # print("Cases calculated: " + str(self.evaluationCount) + "
        print("Calculation time: " + str(
            int((time.time() - startTime) * 1000)) + " ms")
        return move

    """
     alpha: Best AI Move (Max)
     beta: Best Player Move (Min)
     returns: [score, move[0], move[1]]
    """
    @staticmethod
    #@numba.njit(parallel=True)
    def minimax_search_ab(depth, dummy_board_matrix, max_player, alpha, beta):

        # Last depth (terminal node), evaluate the current board_matrix score.
        if depth == 0:
            return [Minimax.evaluate_board_for_white(dummy_board_matrix, not max_player), None, None]

        # Generate all possible moves from this node of the Minimax Tree
        all_possible_moves = Board.generate_moves(dummy_board_matrix)

        # If there are no possible moves left, treat this node as a terminal node and return the score.
        if len(all_possible_moves) == 0:
            return [Minimax.evaluate_board_for_white(dummy_board_matrix, not max_player), None, None]

        best_move = [None, None, None]
        # Generate Minimax Tree and calculate node scores.
        if max_player:
            # Initialize the starting best move with -infinity.
            best_move[0] = -1.0

            # Iterate for all possible moves that can be made.
            for move in all_possible_moves:

                # Play the move on the temporary board_matrix without drawing anything.
                Board.add_stone(dummy_board_matrix, move[1], move[0], False)

                # Call the minimax function for the next depth, to look for a minimum score.
                '''
                // Call the minimax function for the next depth, to look for a minimum score.
                // This function recursively generates new Minimax trees branching from this node 
                // (if the depth > 0) and searches for the minimum white score in each of the sub trees.
                // We will find the maximum score of this depth, among the minimum scores found in the
                // lower depth.
                '''
                temp_move = Minimax.minimax_search_ab(depth - 1, dummy_board_matrix, False, alpha, beta)

                # Backtrack and remove.
                Board.remove_stone(dummy_board_matrix, move[1], move[0])

                # Updating alpha (alpha value holds the maximum score)
                '''
                // Updating alpha (alpha value holds the maximum score)
                // When searching for the minimum, if the score of a node is lower than the alpha 
                // (max score of uncle nodes from one upper level) the whole subtree originating
                // from that node will be discarded, since the maximizing player will choose the 
                // alpha node over any node with a score lower than the alpha. 
                '''
                if temp_move[0] > alpha:
                    alpha = temp_move[0]

                '''
                // Pruning with beta
                // Beta value holds the minimum score among the uncle nodes from one upper level.
                // We need to find a score lower than this beta score, because any score higher than
                // beta will be eliminated by the minimizing player (upper level). If the score is
                // higher than (or equal to) beta, break out of loop discarding any remaining nodes 
                // and/or subtrees and return the last move.
                '''
                if temp_move[0] >= beta:
                    return temp_move

                # Find the move with the maximum score.
                if temp_move[0] > best_move[0]:
                    best_move = temp_move
                    best_move[1] = move[0]
                    best_move[2] = move[1]

        else:
            # Initialize the starting best move using the first move in the list and +infinity score.
            best_move[0] = 100_000_000.0
            best_move[1] = all_possible_moves[0][0]
            best_move[2] = all_possible_moves[0][1]

            # Iterate for all possible moves that can be made.
            for move in all_possible_moves:

                # Play the move on the temporary board_matrix without drawing anything.
                Board.add_stone(dummy_board_matrix, move[1], move[0], True)

                '''
                // Call the minimax function for the next depth, to look for a maximum score.
                // This function recursively generates new Minimax trees branching from this node 
                // (if the depth > 0) and searches for the maximum white score in each of the sub trees.
                // We will find the minimum score of this depth, among the maximum scores found in the
                // lower depth.
                '''
                temp_move = Minimax.minimax_search_ab(depth - 1, dummy_board_matrix, True, alpha, beta)

                Board.remove_stone(dummy_board_matrix, move[1], move[0])

                '''
                // Updating beta (beta value holds the minimum score)
                // When searching for the maximum, if the score of a node is higher than the beta 
                // (min score of uncle nodes from one upper level) the whole subtree originating
                // from that node will be discarded, since the minimizing player will choose the 
                // beta node over any node with a score higher than the beta. 
                '''
                beta = min(temp_move[0], beta)

                '''
                // Pruning with alpha
                // Alpha value holds the maximum score among the uncle nodes from one upper level.
                // We need to find a score higher than this alpha score, because any score lower than
                // alpha will be eliminated by the maximizing player (upper level). If the score is
                // lower than (or equal to) alpha, break out of loop discarding any remaining nodes 
                // and/or subtrees and return the last move.
                '''
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
    def search_winning_move(board_matrix):

        all_possible_moves = Board.generate_moves(board_matrix)
        winning_move = [None, None, None]

        for move in all_possible_moves:
            # Create a temporary board_matrix that is equivalent to the current board_matrix
            dummy_board = Board.clone_matrix(board_matrix)
            # Play the move on that temporary board_matrix without drawing anything
            Board.add_stone(dummy_board, move[1], move[0], False)

            # If the white player has a winning score in that temporary board_matrix, return the move.
            if Minimax.get_score(dummy_board, False, False) >= Minimax.WIN_SCORE:
                winning_move[1] = move[0]
                winning_move[2] = move[1]
                return winning_move

        return None

    # This function calculates the score by evaluating the stone positions in horizontal direction
    @staticmethod
    def evaluate_horizontal(boardMatrix, forBlack: bool, playersTurn: bool):

        evaluations = [0, 2, 0] # [0] -> consecutive count, [1] -> block count, [2] -> score
        '''
        // blocks variable is used to check if a consecutive stone set is blocked by the opponent or
        // the board_matrix border. If the both sides of a consecutive set is blocked, blocks variable will be 2
        // If only a single side is blocked, blocks variable will be 1, and if both sides of the consecutive
        // set is free, blocks count will be 0.
        // By default, first cell in a row is blocked by the left border of the board_matrix.
        // If the first cell is empty, block count will be decremented by 1.
        // If there is another empty cell after a consecutive stones set, block count will again be
        // decremented by 1.
        // 
        Iterate over all rows '''

        for i in range(len(boardMatrix)):
            #// Iterate over all cells in a row
            for j in range( len(boardMatrix[0])):
                #// Check if the selected player has a stone in the current cell
                Minimax.evaluate_directions(boardMatrix, i, j, forBlack, playersTurn, evaluations)

            Minimax.evaluate_directions_after_one_pass(evaluations, forBlack, playersTurn)

        return evaluations[2]

    '''
    This function calculates the score by evaluating the stone positions in vertical direction
     The procedure is the exact same of the horizontal one.
    '''

    @staticmethod
    def evaluate_vertical(boardMatrix, forBlack:bool, playersTurn:bool):
        evaluations = [0, 2, 0] # [0] -> consecutive count, [1] -> block count, [2] -> score

        for j in range(len(boardMatrix[0])):
            for i in range(len(boardMatrix)):
                Minimax.evaluate_directions(boardMatrix, i, j, forBlack, playersTurn, evaluations)
            Minimax.evaluate_directions_after_one_pass(evaluations, forBlack, playersTurn)
        return evaluations[2]

    @staticmethod
    def evaluate_diagonal(boardMatrix, forBlack, playersTurn):
        evaluations = [0, 2, 0]  # [0] -> consecutive count, [1] -> block count, [2] -> score
        # From bottom-left to top-right diagonally
        for k in range(0, 2 * (len(boardMatrix) - 1) + 1):
            iStart = max(0, k - len(boardMatrix) + 1)
            iEnd = min(len(boardMatrix) - 1, k)
            for i in range(iStart, iEnd + 1):
                Minimax.evaluate_directions(boardMatrix, i, k - i, forBlack, playersTurn, evaluations)
            Minimax.evaluate_directions_after_one_pass(evaluations, forBlack, playersTurn)

        # From top-left to bottom-right diagonally
        for k in range(1 - len(boardMatrix), len(boardMatrix)):
            iStart = max(0, k)
            iEnd = min(len(boardMatrix) + k - 1, len(boardMatrix) - 1)
            for i in range(iStart, iEnd + 1):
                Minimax.evaluate_directions(boardMatrix, i, i - k, forBlack, playersTurn, evaluations)
            Minimax.evaluate_directions_after_one_pass(evaluations, forBlack, playersTurn)

        return evaluations[2]

    @staticmethod
    def evaluate_directions(boardMatrix, i, j, isBot, botsTurn, eval):
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

    @staticmethod
    def evaluate_directions_after_one_pass(eval, isBot, playersTurn):
        # End of row, check if there were any consecutive stones before we reached right border
        if eval[0] > 0:
            eval[2] += Minimax.get_consecutive_set_score(eval[0], eval[1], isBot == playersTurn)
        # Reset consecutive stone and blocks count
        eval[0] = 0
        eval[1] = 2


    # This function returns the score of a given consecutive stone set.
    # count: Number of consecutive stones in the set
    # blocks: Number of blocked sides of the set (2: both sides blocked, 1: single side blocked, 0: both sides free)
    @staticmethod
    def get_consecutive_set_score(count, blocks, currentTurn):
        winGuarantee = 1000000

        # If both sides of a set are blocked, this set is worthless return 0 points.
        if blocks == 2 and count < 5:
            return 0

        if count == 5:
            # 5 consecutive wins the game
            return Minimax.WIN_SCORE
        elif count == 4:
            # 4 consecutive stones in the user's turn guarantees a win.
            # (User can win the game by placing the 5th stone after the set)
            if currentTurn:
                return winGuarantee
            else:
                # Opponent's turn
                # If neither side is blocked, 4 consecutive stones guarantees a win in the next turn.
                if blocks == 0:
                    return winGuarantee / 4
                # If only a single side is blocked, 4 consecutive stones limits the opponents move
                # (Opponent can only place a stone that will block the remaining side, otherwise the game is lost
                # in the next turn). So a relatively high score is given for this set.
                else:
                    return 200
        elif count == 3:
            # 3 consecutive stones
            if blocks == 0:
                # Neither side is blocked.
                # If it's the current player's turn, a win is guaranteed in the next 2 turns.
                # (User places another stone to make the set 4 consecutive, opponent can only block one side)
                # However, the opponent may win the game in the next turn, therefore this score is lower than win
                # guaranteed scores but still a very high score.
                if currentTurn:
                    return 50_000
                # If it's the opponent's turn, this set forces the opponent to block one of the sides of the set.
                # So a relatively high score is given for this set.
                else:
                    return 200
            else:
                # One of the sides is blocked.
                # Playmaker scores
                if currentTurn:
                    return 10
                else:
                    return 5
        elif count == 2:
            # 2 consecutive stones
            # Playmaker scores
            if blocks == 0:
                if currentTurn:
                    return 7
                else:
                    return 5
            else:
                return 3
        elif count == 1:
            return 1

        # More than 5 consecutive stones?
        return Minimax.WIN_SCORE * 2

