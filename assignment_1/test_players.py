from random import randint

class RandomPlayer():
    """Player that chooses a move randomly."""    

    def move(self, game, legal_moves, time_left):	
	flag=True
	while flag:
	    if not len(legal_moves[game.__active_players_queen1__]) and not len(legal_moves[game.__active_players_queen1__]): return None,None
	    if len(legal_moves[game.__active_players_queen1__]):
		move1=legal_moves[game.__active_players_queen1__][randint(0,len(legal_moves[game.__active_players_queen1__])-1)]
	    else:	
		move1=None
	
	    if len(legal_moves[game.__active_players_queen2__]):
			move2=legal_moves[game.__active_players_queen2__][randint(0,len(legal_moves[game.__active_players_queen2__])-1)]
	    else:	
		move2=None
	
	    if move1!=move2:
		flag=False
		return move1,move2
	    elif move1==move2 and len(legal_moves[game.__active_players_queen1__])==1 and len(legal_moves[game.__active_players_queen2__])==1:
		return None,None
	    else:
		flag=True

    


class HumanPlayer(): 
    """Player that chooses a move according to
    user's input."""
    def move(self, game, legal_moves, time_left):
        print "here in "
        choice = {}
	print len(legal_moves[game.__active_players_queen1__])
	
        if not len(legal_moves[game.__active_players_queen1__]) and not len(legal_moves[game.__active_players_queen2__]):
	    print "error"
            return None, None
       	i=0
	queen=game.__active_players_queen1__
        for move in legal_moves[game.__active_players_queen1__]:        
            choice.update({i:(queen,move)})
	    print choice
            print('\t'.join(['[%d] q%d: (%d,%d)'%(i,queen,move[0],move[1])] ))
            i=i+1

        j=i
	queen=game.__active_players_queen2__
        for move in legal_moves[game.__active_players_queen2__]:        
            choice.update({j:(queen,move)})
            print('\t'.join(['[%d] q%d: (%d,%d)'%(j,queen,move[0],move[1])] ))
            j=j+1


        valid_choice1 = False
	valid_choice2 = False

        while not valid_choice1 or not valid_choice2:
            try:
                index_queen1 = int(input('Select move index for queen1:'))
                valid_choice1 = 0 <= index_queen1 < i

                if not valid_choice1:
                    print('Illegal move of queen1! Try again.')

		index_queen2 = int(input('Select move index for queen2:'))
                valid_choice2 = i <= index_queen2 < j

                if not valid_choice2:
                    print('Illegal move of queen2! Try again.')
            
            except ValueError:
                print('Invalid index! Try again.')
        
        return choice[index_queen1][1],choice[index_queen2][1]
