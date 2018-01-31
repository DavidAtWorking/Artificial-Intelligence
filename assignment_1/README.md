# Assignment 1 - Isolation

## Setup

Clone this repository:
`git clone https://github.gatech.edu/omscs6601/assignment_1.git`

The submission scripts depend on the presence of 3 python packages - `requests`, `future`, and `nelson`. Install them using the command below:

`pip install -r requirements.txt`

Read [setup.md](./setup.md) for more information on how to effectively manage your git repository and troubleshooting information.

## Overview

This assignment will cover some of the concepts discussed in the Adversarial Search lectures. You will be implementing game playing agents for a variant of the game Isolation.

### The Game


The rules of 2 Queen's Isolation are simple. There are two players, four game pieces and a 7-by-7 grid of squares. At the beginning of the game, the first player places both the pieces on any two different squares. From that point on, the players alternate turns moving both the pieces like a Queen in chess (any number of open squares vertically, horizontally, or diagonally). When the piece is moved, the square that was previously occupied is blocked. That square can not be used for the remainder of the game. The piece can not move through blocked squares. The first player who is unable to move any one of the queens loses.



### The Files

While you'll only have to edit and submit `player_submission.py`, there are a number of notable files:

1. `isolation.py`: Includes the `Board` class and a function for printing out a game as text. Avoid changing contents of this file. We have same file on the server's side.
2. `player_submission.py`: Where you'll implement the required methods for your agents.
3. `player_submission_tests.py`: Sample tests to validate your agents locally.
3. `test_players.py`: Example agents used to play isolation locally.
4. `submit.py`: Script to submit your work to evaluate against the first 2 tests (mentioned in the next section).
5. `submit_a.py`: Script to submit your work to evaluate against the middle 2 tests (mentioned in the next section).
6. `submit_b.py`: Script to submit your work to evaluate against the last 2 tests (mentioned in the next section).

### The Assignment

Your task is to create an AI that can play and win a game of 2 Queen's Isolation. Your AI will be tested against several pre-baked AIs as well as your peers’ AI systems. You will implement your AI in Python 2.7, using our provided code as a starting point.

In this repository, we provide:

- A class for representing the game state
- A function for printing the game board
- A function for generating legal game states
- A class for running unit tests
- A random AI (baseline test)

Your goal is to implement the following parts of the AI in the class CustomPlayer:

1. Evaluation functions (`OpenMoveEvalFn()` and `CustomEvalFn()`)
2. The minimax algorithm (`minimax()`)
3. Alpha-beta pruning (`alphabeta()`)

Your agent will have a limited amount of time to act each turn (10 seconds). We will call these functions directly so **don’t modify** the <u>function names</u> or the <u>parameters</u>.

In addition to checking time each turn, you will be penalized if your agent takes more than a few minutes at construction time (for example, if you attempt to load the entire set of possible board states from memory). We have divided the tests into three(mentioned in details in next grading section below).  In total, your submission will be allowed to run for a maximum of <u>120 minutes</u> before being interrupted <u> for second and third section. Your submission will be allowed to run for a maximum of <u>30 minutes</u> for first section. 

These are the bare minimum requirements for your AI, and the rest is up to you. You will be scored according to how well your AI performs against some baseline AIs that we provide (see “Grading”). If you want to improve over the base performance, here are a few suggestions:

- Use partition techniques.
- Store the evaluation scores for past moves.
- Modify your evaluation function to account for “killer moves”.
- Optimize functions that are called often.
- Order nodes to maximize pruning.

### Grading

The grade you receive for the assignment will be determined as follows:

| Points    | Condition                                |
| --------- | ---------------------------------------- |
| 5 points | You write an evaluation function, OpenMoveEval, which returns the number of moves that the AI minus the number of moves opponent can make, and your evaluation function performs correctly on some sample boards we provide. |
| 30 points | Your AI defeats a random player >= 90% of the time. |
| 20 points | Your AI defeats an agent with OpenMoveEval function that uses minimax to level 2  >= 65% of the times. |
| 20 points | Your AI defeats an agent with OpenMoveEval function that uses alphabeta to level 3  >= 65% of the times. |
| 20 points | Your AI defeats an agent with OpenMoveEval function that uses iterative deepening and alpha-beta pruning >= 65% of the time. |
| 5 points | Your AI defeats an agent with Kshitish's secret evaluation function that uses iterative deepening and alpha-beta pruning and optimizes various aspects of the game player >= 85% of the time  |

We have divided the tests into three. The first section contains tests for the first two parts, the second section has tests for the middle two parts and the third section has tests for last two parts.
Submission policy: One submission per 120 minutes for second and third section separately. One submission per 30 minutes for first section (OpenMoveEvalFn and RandomPlayer).  Grades will be based on the last submission made per section. (We are running our largest class to date, so we reserve the right to modify these rules depending upon the load on the servers).

### Botfight! (Extra Credit)

In addition to the basic assignment, you will have the option to compete against your peers for the glory of being the Spring 2018 AI-Game-Playing champ. We’ll set up a system to pit your AI against others, and we’ll be handing out extra credit for the top players. May the odds be ever in your favor.

If you wish to compete in the tournament, simply include a plaintext file with a description of your agent, titled ‘AI.txt’, while submitting for the third section of tests (submit_b) and your CustomPlayer instance will be enlisted.

If you compete in the AI tournament and your agent finishes in the top 10, you will receive a bonus points for this assignment (bonus points are added to the grades of each assignment. Not to final score. ):

- Best Overall:  12 bonus points added to the assignment score.
- Second Best: 10 bonus points.
- Third Best: 7 bonus points.
- Fourth to Tenth Best: 5 bonus point.

## How to Submit Your Code

A friendly reminder: please ensure that your submission is in `player_submission.py`. The scripts described in the following section automatically send that file to the servers for processing.

To submit your code and have it evaluated for a grade for first section, use `python submit.py`, for evaluation of second section use `python submit_a.py` and for third section use `python submit_b.py`.  Ensure that you have created the required AI.txt to enter the tournament.

## Your Classes (`player_submission.py`)

As discussed above, the follow classes must be populated in `player_submission.py`:

1. `OpenMoveEvalFn`
2. `CustomEvalFn`
3. `CustomPlayer`

Aside from those 3 classes, any added code (such as additional tests) **MUST** be included in **`player_submissions_tests.py`** or another separate file. We need to be certain that code outside of these classes does not get accidentally executed.

### Evaluation Functions

These functions will inform the value judgements your AI will make when choosing moves. There are 2 classes:

- `OpenMoveEvalFn` -Returns the number of available moves open for the your player minus the number of moves available for opponent player(consider overlapping moves of your own queens only once). All baseline tests will use this function. **This is mandatory**
- `CustomEvalFn` - You are encouraged to create your own evaluation function here.

**DO** submit the code within this class (and only the code within this class).

#### Tips

1. You may write additional code within each class. However, we will only be invoking the `score()` function. You may not change the signature of this function.
2. When writing additional code to test, try to do so in separate classes (do not use ours). It allows for independent test execution and you can be sure that *all* the code within the EvalFn cells belong only to the EvalFn classes

```python

class OpenMoveEvalFn:
    def score(self, game, maximizing_player_turn=True):
        """Score the current game state        
        Evaluation function that outputs a score equal to how many 
        moves are open for AI player on the board minus how many moves 
	are open for Opponent's player on the board.
        """
	# TODO: finish this function!
        raise NotImplementedError



class CustomEvalFn:
    def score(self, game, maximizing_player_turn=True):
        """Score the current game state        
        Custom evaluation function that acts however you think it should. This 
        is not required but highly encouraged if you want to build the best 
        AI possible.
        """
        # TODO: finish this function!
        raise NotImplementedError

```

### `CustomPlayer`

This is the meat of the assignment. A few notes about the class:

- You are not permitted to change the function signatures of any of the provided methods.
- You are permitted to change the default values within the function signatures provided. In fact, when you have your custom evaluation function, you are encouraged to change the default values for `__init__` to use the new eval function.
- You are free change the contents of each of the provided methods. When you are ready with `alphabeta()`, for example, you are encouraged to update `move()` to use that function instead.
- You are free to add more methods to the class.
- You may not create additional external functions and classes that are referenced from within this class.

**DO** submit the code within this class (and only the code within this class).

```python
class CustomPlayer:
    # TODO: finish this class!
    """Player that chooses a move using 
    your evaluation function and 
    a minimax algorithm 
    with alpha-beta pruning.
    You must finish and test this player
    to make sure it properly uses minimax
    and alpha-beta to return a good move."""

    def __init__(self, search_depth=3, eval_fn=OpenMoveEvalFn()):
        """Initializes your player.
        
        if you find yourself with a superior eval function, update the default 
        value of `eval_fn` to `CustomEvalFn()`
        
        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Utility function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth

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

        best_move_queen1,best_move_queen2, utility = self.minimax(game, time_left, depth=self.search_depth)	
        return best_move_queen1,best_move_queen2

    def utility(self, game, maximizing_player):
        """Can be updated if desired. Not compulsory. """
        return self.eval_fn.score(game)

    def minimax(self, game, time_left, depth=2, maximizing_player=True):
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
        raise NotImplementedError
        return best_move_queen1,best_move_queen2, best_val

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
        raise NotImplementedError
	return best_move_queen1,best_move_queen2, val

```

### Built-in Tests

In `player_submission_tests.py` several built-in tests can be found in the **`main()`** function. We've included these to help you test your player and evaluation function as well as to give you an idea of how the classes are used. Feel free to play around with the code and add more tests.

## Helper Player classes (`test_players.py`)

We include 2 player types for you to test against locally:

- `RandomPlayer` - chooses a legal move randomly from among the available legal moves
- `HumanPlayer` - allows *YOU* to play against the AI

**DO NOT** submit these two players. You are however free to change these classes as you see fit. Know that any changes you make will be solely for the benefit of your own tests.

