#!/usr/bin/env python
from isolation import Board, game_as_text
from random import randint
import random
# This file is your main submission that will be graded against. Do not
# add any classes or functions to this file that are not part of the classes
# that we want.


class OpenMoveEvalFn:

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state
        
        Evaluation function that outputs a score equal to how many 
        moves are open for AI player on the board minus how many moves 
	are open for Opponent's player on the board.

	Note:
		1. Be very careful while doing opponent's moves. You might end up 
		   reducing your own moves.
		2. Here if you add overlapping moves of both queens, you are considering one available square twice.
		   Consider overlapping square only once. In both cases- myMoves and in OppMoves. 
		3. If you think of better evaluation function, do it in CustomEvalFn below. 
            
        Args
            param1 (Board): The board and game state.
            param2 (bool): True if maximizing player is active.

        Returns:
            float: The current state's score. MyMoves-OppMoves.
            
        """

	# TODO: finish this function!
        # raise NotImplementedError
	legal_moves_dict    = game.get_legal_moves()
	opponent_moves_dict  = game.get_opponent_moves()
	legal_set_1,legal_set_2 = map(set, legal_moves_dict.values())
	opponent_set_1,opponent_set_2 = map(set, opponent_moves_dict.values())
        my_moves = len(legal_set_1) + len(legal_set_2) - len(legal_set_1 & legal_set_2)
        oppo_moves = len(opponent_set_1) + len(opponent_set_2) - len(opponent_set_1 & opponent_set_2)
         
        return my_moves - oppo_moves if maximizing_player_turn else oppo_moves - my_moves
	    


class CustomEvalFn:

    def __init__(self):
        pass

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state
        
        Custom evaluation function that acts however you think it should. This 
        is not required but highly encouraged if you want to build the best 
        AI possible.
        
        Args
            game (Board): The board and game state.
            maximizing_player_turn (bool): True if maximizing player is active.

        Returns:
            float: The current state's score, based on your own heuristic.
            
        """

        # TODO: finish this function!
        # raise NotImplementedError
	legal_moves_dict    = game.get_legal_moves()
	opponent_moves_dict  = game.get_opponent_moves()
	legal_set_1,legal_set_2 = map(set, legal_moves_dict.values())
	opponent_set_1,opponent_set_2 = map(set, opponent_moves_dict.values())
	if maximizing_player_turn:
	    my_moves = len(legal_set_1) + len(legal_set_2) - len(legal_set_1 & legal_set_2)
	    oppo_moves = len(opponent_set_1) + len(opponent_set_2) - len(opponent_set_1 & opponent_set_2)
	else:
	    oppo_moves = len(legal_set_1) + len(legal_set_2) - len(legal_set_1 & legal_set_2)
	    my_moves = len(opponent_set_1) + len(opponent_set_2) - len(opponent_set_1 & opponent_set_2)
	if game.move_count < 3:
	    return my_moves
	elif my_moves==0 and oppo_moves!=0:
	    return float('-inf')
	elif my_moves!=0 and oppo_moves==0:
	    return float('inf')
	elif my_moves==0 and oppo_moves==0:
	    return -5
	else:
	    return my_moves - 0.5 * oppo_moves	
        

class CustomPlayer:
    # TODO: finish this class!
    """Player that chooses a move using 
    your evaluation function and 
    a minimax algorithm 
    with alpha-beta pruning.
    You must finish and test this player
    to make sure it properly uses minimax
    and alpha-beta to return a good move."""

    def __init__(self, search_depth=3, eval_fn=CustomEvalFn()):
        """Initializes your player.
        
        if you find yourself with a superior eval function, update the default 
        value of `eval_fn` to `CustomEvalFn()`
        
        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Utility function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth
	self.is_first_move = True
	self.opening_moves = [(6,6), (5,5), (2,0), (0,6), (6,0), (0,4), (0,5), (1,0), (0,1)]
	self.is_top_search = True
	self.is_second_player = False
	

    def move(self, game, legal_moves, time_left):
	"""Called to determine one move by your agent
        
	Note:
		1. Do NOT change the name of this 'move' function. We are going to call 
		the this function directly. 
		2. Change the name of minimax function to alphabeta function when 
		required. Here we are talking about 'minimax' function call,
		NOT 'move' function name.

        Args:
            game (Board): The board and game state.
            legal_moves (dict): Dictionary of legal moves and their outcomes
            time_left (function): Used to determine time left before timeout
            
        Returns:
            (tuple, tuple): best_move_queen1, best_move_queen2
        """
	self.is_top_search = True
	if self.is_first_move and game.__last_queen_move__.values() == [(-1,-1)]*4:
	    self.is_second_player = True
	    self.is_first_move = False
	    moves_queen1 = [x for x in self.opening_moves if x in legal_moves[11]]	    
	    moves_queen2 = [x for x in self.opening_moves if x in legal_moves[12]]
	    for move1 in moves_queen1:
		for move2 in [x for x in moves_queen2 if x != move1]:
		    return move1, move2
	    #for move1, move2 in zip(moves_queen1, moves_queen2):
	    #	if move1 != move2:
	    #	    return move1, move2
	    #return moves_queen1[0], moves_queen2[1]
	else:
	    depth = self.search_depth
	    best_move_queen1, best_move_queen2, best_val = self.alphabeta(game, time_left, depth)
	    #if game.move_count <= 2:
		#return best_move_queen1, best_move_queen2
	    while self.is_time_left(time_left) and depth < 100:
		self.is_top_search = True
		depth += 1
		move_queen1, move_queen2, val = self.alphabeta(game, time_left, depth)
		best_move_queen1, best_move_queen2 = move_queen1, move_queen2
		#if val > best_val:
		#    best_move_queen1, best_move_queen2 = move_queen1, move_queen2
        return best_move_queen1,best_move_queen2


    def utility(self, game, maximizing_player):
        """Can be updated if desired. Not compulsory. """
        return self.eval_fn.score(game, maximizing_player)

    def is_time_left(self, time_left):
	return time_left() > 500

    def minimax(self, game, time_left, depth=3, maximizing_player=True):
        """Implementation of the minimax algorithm
        
        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple,tuple, int): best_move_queen1,best_move_queen2, val
        """
	# TODO: finish this function!
        # raise NotImplementedError
	flag = False
	dic = game.get_legal_moves()
	legal_moves_queen1, legal_moves_queen2 = dic.values()[0], dic.values()[1]
	if self.is_top_search and self.is_second_player:
	    self.is_top_search = False
	    random.shuffle(legal_moves_queen1)
	    random.shuffle(legal_moves_queen2)
	# edge condition
	if depth == 0 or not legal_moves_queen1 or not legal_moves_queen2:
	    return (-1,-1),(-1,-1),self.utility(game, maximizing_player)
	# initialization
	best_move_queen1, best_move_queen2 = legal_moves_queen1[0], legal_moves_queen2[0]
	best_val = float("-inf") if maximizing_player else float("inf")
	# non edge condition
	for move_queen1 in legal_moves_queen1:
	    for move_queen2 in [x for x in legal_moves_queen2 if x != move_queen1]:
	#for move_queen1, move_queen2 in zip(legal_moves_queen1, legal_moves_queen2):
		current_val = self.minimax(game.forecast_move(move_queen1, move_queen2), time_left, depth-1, not maximizing_player)[2]
		if (maximizing_player and current_val > best_val) or (not maximizing_player and current_val < best_val):
		    best_move_queen1, best_move_queen2 = move_queen1, move_queen2
		    best_val = current_val
		if not self.is_time_left(time_left):
		    flag = True
		    break
	    if flag:
		break
        return best_move_queen1, best_move_queen2, best_val

    def alphabeta(self, game, time_left, depth=3, alpha=float("-inf"), beta=float("inf"),maximizing_player=True):
        """Implementation of the alphabeta algorithm
        
        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple,tuple, int): best_move_queen1,best_move_queen2, val
        """
        # TODO: finish this function!
        # raise NotImplementedError
	flag = False
	dic = game.get_legal_moves()
	legal_moves_queen1, legal_moves_queen2 = dic.values()[0], dic.values()[1]
	if self.is_top_search and self.is_second_player:
	    self.is_top_search = False
	    random.shuffle(legal_moves_queen1)
	    random.shuffle(legal_moves_queen2)
	if depth == 0 or not legal_moves_queen1 or not legal_moves_queen2:
	    return (-1,-1),(-1,-1),self.utility(game, maximizing_player)
	best_move_queen1, best_move_queen2 = legal_moves_queen1[0], legal_moves_queen2[0]
	best_val = float("-inf") if maximizing_player else float("inf")
	#for move_queen1, move_queen2 in zip(legal_moves_queen1, legal_moves_queen2):
	for move_queen1 in legal_moves_queen1:
	    for move_queen2 in [x for x in legal_moves_queen2 if x != move_queen1]:
		current_val = self.alphabeta(game.forecast_move(move_queen1, move_queen2), time_left, depth-1, alpha, beta, not maximizing_player)[2]
		if (maximizing_player and current_val > best_val) or (not maximizing_player and current_val < best_val):
		    best_move_queen1, best_move_queen2 = move_queen1, move_queen2
		    best_val = current_val
		if maximizing_player:
		    alpha	= max(alpha, current_val)
		else:
		    beta	= min(beta, current_val)
		if alpha > beta:
		    break
		if not self.is_time_left(time_left):
		    flag = True
		    break
	    if flag:
		break
	return best_move_queen1,best_move_queen2, best_val

