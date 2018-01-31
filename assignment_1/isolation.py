from copy import deepcopy
from time import time, sleep
import platform
import random
#import io
import StringIO
#import resource
if platform.system() != 'Windows':
    import resource

import sys,os
sys.path[0] = os.getcwd()

class Board:
    BLANK = 0
    NOT_MOVED = (-1, -1)

    __active_players_queen1__= None                       
    __inactive_players_queen1__= None
    __active_players_queen2__= None                        
    __inactive_players_queen2__= None
    
    
    def __init__(self, player_1, player_2, width=7, height=7):
        self.width=width
        self.height=height
        
        self.queen_11 = "queen11"
        self.queen_12 = "queen12"
        self.queen_21 = "queen21"
        self.queen_22 = "queen22"
        
        self.__board_state__ = [ [Board.BLANK for i in range(0, width)] for j in range(0, height)]
        self.__last_queen_move__ = {self.queen_11:Board.NOT_MOVED, self.queen_12:Board.NOT_MOVED, self.queen_21:Board.NOT_MOVED, self.queen_22:Board.NOT_MOVED}
        self.__queen_symbols__ = {Board.BLANK: Board.BLANK, self.queen_11:11, self.queen_12:12, self.queen_21:21, self.queen_22:22}     
        
        self.move_count = 0
        
        self.__queen_11__ = self.queen_11
        self.__queen_12__ = self.queen_12
        self.__queen_21__ = self.queen_21
        self.__queen_22__ = self.queen_22
        
        
        self.__player_1__ = player_1
        self.__player_2__ = player_2
        
        
        self.__active_player__ = player_1
        self.__inactive_player__ = player_2 
        
        self.__active_players_queen1__= 11                    
        self.__active_players_queen2__= 12
        self.__inactive_players_queen1__= 21 
        self.__inactive_players_queen2__= 22 
        
        
    def get_queen_name(self, queen_num):
	#returns name of queen. 
        if queen_num == 11:
            return self.queen_11
        elif queen_num ==12:
            return self.queen_12
        elif queen_num == 21:
            return self.queen_21
        elif queen_num ==22:
            return self.queen_22
        else :
            return None
            
    
    def get_state(self):
        return deepcopy(self.__board_state__)
   
        
    def __apply_move__(self, move_queen1, move_queen2):
	#apply move of queen1 of active player

        row,col = move_queen1
	self.__last_queen_move__[self.get_queen_name(self.__active_players_queen1__)] = move_queen1     
        self.__board_state__[row][col] = self.__queen_symbols__[self.get_queen_name(self.__active_players_queen1__)]  

	#apply move of queen2 of active player
	row,col = move_queen2	
        self.__last_queen_move__[self.get_queen_name(self.__active_players_queen2__)] = move_queen2     
        self.__board_state__[row][col] = self.__queen_symbols__[self.get_queen_name(self.__active_players_queen2__)]   
        
        #swap the players        
        tmp = self.__active_player__
        self.__active_player__ = self.__inactive_player__
        self.__inactive_player__ = tmp
        
        #swaping the queens        
        tmp = self.__active_players_queen1__
        self.__active_players_queen1__ = self.__inactive_players_queen1__
        self.__inactive_players_queen1__ = tmp        
        tmp = self.__active_players_queen2__
        self.__active_players_queen2__ = self.__inactive_players_queen2__
        self.__inactive_players_queen2__ = tmp
        
	#increment move count
        self.move_count = self.move_count + 1
        
        



    def copy(self):
        b = Board(self.__player_1__, self.__player_2__, width=self.width, height=self.height)
        for key, value in self.__last_queen_move__.items():
            b.__last_queen_move__[key] = value
        for key, value in self.__queen_symbols__.items():
            b.__queen_symbols__[key] = value
        b.move_count = self.move_count
        b.__active_player__ = self.__active_player__
        b.__inactive_player__ = self.__inactive_player__
        b.__active_players_queen1__ = self.__active_players_queen1__
        b.__active_players_queen2__ = self.__active_players_queen2__
        b.__inactive_players_queen1__ = self.__inactive_players_queen1__
        b.__inactive_players_queen2__ = self.__inactive_players_queen2__
        b.__board_state__ = self.get_state()
        return b



    def forecast_move(self, move_queen1, move_queen2):    
        new_board = self.copy()
        new_board.__apply_move__(move_queen1, move_queen2)
        return new_board

    def get_active_player(self):
        return self.__active_player__    

    def get_inactive_player(self):
        return self.__inactive_player__    
    
    def get_active_players_queen(self):
        return self.__active_players_queen1__, self.__active_players_queen2__
    
    def get_inactive_players_queen(self):
        return self.__inactive_players_queen1__, self.__inactive_players_queen2__

    def get_opponent_moves(self):      
        move_by_q1 = self.__last_queen_move__[self.get_queen_name(self.__inactive_players_queen1__)]
        move_by_q2 = self.__last_queen_move__[self.get_queen_name(self.__inactive_players_queen2__)]        
        return {self.__inactive_players_queen1__:self.__get_moves__(move_by_q1) , self.__inactive_players_queen2__:self.__get_moves__(move_by_q2)}
    
    def get_legal_moves(self):
        move_by_q1 = self.__last_queen_move__[self.get_queen_name(self.__active_players_queen1__)]
        move_by_q2 = self.__last_queen_move__[self.get_queen_name(self.__active_players_queen2__)]
        return {self.__active_players_queen1__:self.__get_moves__(move_by_q1) , self.__active_players_queen2__:self.__get_moves__(move_by_q2)}

    def get_legal_moves_of_queen1(self):
        return self.__get_moves__(self.__last_queen_move__[self.get_queen_name(self.__active_players_queen1__)])

    def get_legal_moves_of_queen2(self):
        return self.__get_moves__(self.__last_queen_move__[self.get_queen_name(self.__active_players_queen2__)])

    def __get_moves__(self, move):
               
        if move == self.NOT_MOVED:
            return self.get_first_moves()
        if self.move_count < 1:
            return self.get_first_moves()

        r, c = move

        directions = [ (-1, -1), (-1, 0), (-1, 1),
                        (0, -1),          (0,  1),
                        (1, -1), (1,  0), (1,  1)]

        fringe = [((r+dr,c+dc), (dr,dc)) for dr, dc in directions 
                if self.move_is_legal(r+dr, c+dc)]

        valid_moves = []

        while fringe:
            move, delta = fringe.pop()
            
            r, c = move
            dr, dc = delta

            if self.move_is_legal(r,c):
                new_move = ((r+dr, c+dc), (dr,dc))
                fringe.append(new_move)
                valid_moves.append(move)

        return valid_moves



    def get_first_moves(self):
        return [ (i,j) for i in range(0,self.height) for j in range(0,self.width) if self.__board_state__[i][j] == Board.BLANK]

    def move_is_legal(self, row, col):
        return 0 <= row < self.height and \
               0 <= col < self.width  and \
                self.__board_state__[row][col] == Board.BLANK

    def print_board(self):

	p11_r, p11_c = self.__last_queen_move__[self.__queen_11__]
	p12_r, p12_c = self.__last_queen_move__[self.__queen_12__]
        p21_r, p21_c = self.__last_queen_move__[self.__queen_21__]
        p22_r, p22_c = self.__last_queen_move__[self.__queen_22__]
        b = self.__board_state__

        out = ''

        for i in range(0, len(b)):
            for j in range(0, len(b[i])):
                if not b[i][j]:
                    out += '  '

                elif i == p11_r and j == p11_c:
                    out += '11'
                elif i == p12_r and j == p12_c:
                    out += '12'
                elif i == p21_r and j == p21_c:
                    out += '21'
                elif i == p22_r and j == p22_c:
                    out += '22'
                else:
                    out += '--'

                out += ' | '
            out += '\n\r'

        return out


    def play_isolation(self, time_limit = 10000, print_moves=False):

        move_history = []
        mi=1
        
        if platform.system() == 'Windows':
            def curr_time_millis():
                return int(round(time() * 1000))
        else:
            def curr_time_millis():
                return 1000 * resource.getrusage(resource.RUSAGE_SELF).ru_utime

        while True:
            game_copy = self.copy()            
            move_start = curr_time_millis()
            time_left = lambda : time_limit - (curr_time_millis() - move_start)            
            curr_move = Board.NOT_MOVED

            try:
                legal_player_moves=self.get_legal_moves()
		curr_move_queen1, curr_move_queen2 = self.__active_player__.move(game_copy,legal_player_moves , time_left) #queen added in return 
                 
            except AttributeError as e:
                raise e
            except Exception as e:
                print e
                pass
            
            if curr_move_queen1 is None or curr_move_queen2 is None:
                if print_moves:
                    print 'Winner: ' + str(self.__inactive_player__)
		return  self.__inactive_player__, move_history, "Move of this player is None"
	
	    if curr_move_queen1 == curr_move_queen2:
		if print_moves:
                    print 'Winner: ' + str(self.__inactive_player__)
		return  self.__inactive_player__, move_history, "Queen1 move same as Queen2 move. Not allowed!"

            if self.__active_player__ == self.__player_1__:
                move_history.append([[curr_move_queen1,curr_move_queen2]])
            else:
		move_history[-1].append([curr_move_queen1,curr_move_queen2])
                
            if time_limit and time_left() <= 0:     
		if print_moves:
                    print 'Winner: ' + str(self.__inactive_player__)           
                return  self.__inactive_player__, move_history, "timeout"
            
            legal_moves_of_queen1 =  self.get_legal_moves_of_queen1()
            legal_moves_of_queen2 =  self.get_legal_moves_of_queen2()
            
            if curr_move_queen1 not in legal_moves_of_queen1:
		if print_moves:
                    print 'Winner: ' + str(self.__inactive_player__)
                return self.__inactive_player__, move_history, "illegal move queen1"
            
            if curr_move_queen2 not in legal_moves_of_queen2:
		if print_moves:
                    print 'Winner: ' + str(self.__inactive_player__)
                return self.__inactive_player__, move_history, "illegal move queen2"
            
            if curr_move_queen1 not in legal_moves_of_queen1 and curr_move_queen2 not in legal_moves_of_queen2:     
		if print_moves:
                    print 'Winner: ' + str(self.__inactive_player__)           
                return self.__inactive_player__, move_history, "illegal move both queens"
            
            self.__apply_move__(curr_move_queen1, curr_move_queen2)


    def __apply_move_write__(self, move_queen1,move_queen2):
        row,col = move_queen1
        self.__last_queen_move__[self.get_queen_name(self.__active_players_queen1__)] = move_queen1    
        self.__board_state__[row][col] = self.__queen_symbols__[self.get_queen_name(self.__active_players_queen1__)] 
	
	row,col = move_queen2
        self.__last_queen_move__[self.get_queen_name(self.__active_players_queen2__)] = move_queen2    
        self.__board_state__[row][col] = self.__queen_symbols__[self.get_queen_name(self.__active_players_queen2__)]   
        
        #swap the players        
        tmp = self.__active_player__
        self.__active_player__ = self.__inactive_player__
        self.__inactive_player__ = tmp

	#swaping the queens        
        tmp = self.__active_players_queen1__
        self.__active_players_queen1__ = self.__inactive_players_queen1__
        self.__inactive_players_queen1__ = tmp
        
        tmp = self.__active_players_queen2__
        self.__active_players_queen2__ = self.__inactive_players_queen2__
        self.__inactive_players_queen2__ = tmp

        self.move_count = self.move_count + 1

def game_as_text(winner, move_history,  termination="", board=Board(1,2)):
    print(winner)
    #ans = io.StringIO()
    ans = StringIO.StringIO()
    k=0
    #print move_history
    for i, move1 in enumerate(move_history):
        p1_move = move1[0]
	
        ans.write("  player1 "+"%d." % i + "Queen1: (%d,%d)" % p1_move[0] + "Queen2: (%d,%d)\r\n" % p1_move[1])
        if p1_move != Board.NOT_MOVED:
            board.__apply_move_write__(p1_move[0], p1_move[1])
        ans.write(board.print_board())

        if len(move1) > 1:
            p2_move = move1[1]
            ans.write("  player2 "+"%d." % i + "Queen1: (%d,%d)" % p2_move[0] + "Queen2: (%d,%d)\r\n" % p2_move[1])
            if p2_move != Board.NOT_MOVED:
                board.__apply_move_write__(p2_move[0] ,p2_move[1] )
            ans.write(board.print_board())
        k=k+1
    ans.write(termination + "\r\n")
    ans.write("Winner: " + str(winner) + "\r\n")

    return ans.getvalue()

if __name__ == '__main__':

    print("Starting game:")

    from test_players import RandomPlayer
    from test_players import HumanPlayer

    board = Board(RandomPlayer(), HumanPlayer())
    winner, move_history, termination = board.play_isolation(time_limit=30000, print_moves=True)
    print game_as_text(winner, move_history, termination, board_copy)

