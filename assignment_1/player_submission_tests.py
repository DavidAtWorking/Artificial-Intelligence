#!/usr/bin/env python
import traceback
from player_submission import OpenMoveEvalFn, CustomEvalFn, CustomPlayer
from isolation import Board, game_as_text
from test_players import RandomPlayer, HumanPlayer
import resource
from time import time, sleep

def main():

    try:
        sample_board = Board(RandomPlayer(), RandomPlayer())
        # setting up the board as though we've been playing
        sample_board.move_count = 4
        sample_board.__board_state__ = [
            [11, 0, 0, 0, 21, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 22, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 12, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]
        sample_board.__last_queen_move__ = {sample_board.queen_11: (0, 0), sample_board.queen_12: (4, 5),
                                            sample_board.queen_21: (0, 4), sample_board.queen_22: (2, 2)}
        test = sample_board.get_legal_moves()
        h = OpenMoveEvalFn()
        print 'OpenMoveEvalFn Test: This board has a score of %s.' % (h.score(sample_board))
    except NotImplementedError:
        print 'OpenMoveEvalFn Test: Not implemented'
    except:
        print 'OpenMoveEvalFn Test: ERROR OCCURRED'
        print traceback.format_exc()

    try:
        """Example test to make sure
        your minimax works, using the
        OpenMoveEvalFunction evaluation function.
	This can be used for debugging your code
	with different model Board states. 
	Especially important to check alphabeta 
	pruning"""
        # create dummy 5x5 board

        p1 = RandomPlayer()	
	p2 = HumanPlayer()
        b = Board(p1, p2, 5, 5)
	
        b.__board_state__ = [
            [0, 0 , 0, 0, 0],
            [0, 0,  0, 22, 0],
            [0, 0,  0, 11, 0],
            [0, 0,  0, 21, 12],
            [0, 0 , 0, 0, 0]
        ]
        b.__last_queen_move__["queen11"] = (2, 3)
        b.__last_queen_move__["queen12"] = (3, 4)
        b.__last_queen_move__["queen21"] = (3, 3)
        b.__last_queen_move__["queen22"] = (1, 3)
        b.move_count = 4

        output_b = b.copy()
	legal_moves=b.get_legal_moves()
        winner, move_history,  termination = b.play_isolation()
        print 'Minimax Test: Runs Successfully'
        # Uncomment to see example game
        print game_as_text(winner, move_history, termination, output_b)
    except NotImplementedError:
        print 'Minimax Test: Not Implemented'
    except:
        print 'Minimax Test: ERROR OCCURRED'
        print traceback.format_exc()

       

    """Example test you can run
    to make sure your AI does better
    than random."""    
    try:
	r = RandomPlayer()
	h = CustomPlayer()
	game = Board(r, h, 7, 7)
	output_b = game.copy()
	winner, move_history, termination = game.play_isolation()
	print game_as_text(winner, move_history, termination, output_b)
	if 'CustomPlayer' in str(winner):
	    print 'CustomPlayer Test: CustomPlayer Won'	   
	else:
	    print 'CustomPlayer Test: CustomPlayer Lost'
	# Uncomment to see game
	# print game_as_text(winner, move_history, termination, output_b)
    except NotImplementedError:
	print 'CustomPlayer Test: Not Implemented'
    except:
	print 'CustomPlayer Test: ERROR OCCURRED'
	print traceback.format_exc()
	    
   
    
if __name__ == "__main__":
    main()
