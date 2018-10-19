# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        pacmanPosition = successorGameState.getPacmanPosition()
        score = successorGameState.getScore()

        #if PacMan would occupy the same space as a ghost, add reward if ghost is scared, and add penalty if ghost is not
        for ghostState in newGhostStates:
            if ghostState.getPosition() == pacmanPosition:
                if ghostState.scaredTimer is 0:
                    score += -100
                else:
                    score += 100

        #add penalty to STOP action so PacMan will keep moving
        if action == Directions.STOP:
            score += -100

        #add reward to reaching Capsule
        if pacmanPosition in successorGameState.getCapsules():
            score += 100

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        #Get all possible legal actions for PacMan
        pacmanActions = gameState.getLegalActions(0)

        #Get successor states for all PacMan actions
        successorStates = [gameState.generateSuccessor(0, action) for action in pacmanActions]

        #Uses minimizer to get scores, and then return the best score (borrowed code from Reflex Agent)
        scores = [self.minimumFunction(0, state, 1) for state in successorStates]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return pacmanActions[chosenIndex]

    #Minimum function
    #If the gamestate is win/lose, or if self.depth is equal to the current depth passed in, return the gamestate
    #Minimum function gets all of the legal actions for PacMan and all of the resulting states from those actions
    #It then passes each state to the maximum function when the agent is a ghost
    def minimumFunction(self, currentDepth, gameState, agentIndex):
        if(self.depth == currentDepth or gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)
        minMoves = gameState.getLegalActions(agentIndex)
        resultStates = [gameState.generateSuccessor(agentIndex, action) for action in minMoves]

        if (agentIndex >= gameState.getNumAgents() - 1):
            score = [self.maximumFunction(currentDepth + 1, state, agentIndex) for state in resultStates]
        else:
            score = [self.minimumFunction(currentDepth, state, agentIndex + 1) for state in resultStates]
        return min(score)

    #Maximum function
    #If the gamestate is win/lose, or if self.depth is equal to the current depth passed in, return the gamestate
    #Maximum function gets all of the legal actions for PacMan and all of the resulting states from those actions
    #It then passes each state to the minimum function and returns the maximum score
    def maximumFunction(self, currentDepth, gameState, agentIndex):
        if(self.depth == currentDepth or gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)
        maxMoves = gameState.getLegalActions(0)
        resultStates = [gameState.generateSuccessor(0, action) for action in maxMoves]
        score = [self.minimumFunction(currentDepth, state, 1) for state in resultStates]
        return max(score)

        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        # Get all possible legal actions for PacMan
        pacmanActions = gameState.getLegalActions(0)

        # Get successor states for all PacMan actions
        successorStates = [gameState.generateSuccessor(0, action) for action in pacmanActions]

        # Uses minimizer to get scores, and then return the best score (borrowed code from Reflex Agent)
        scores = [self.expectimaxFunction(0, state, 1) for state in successorStates]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return pacmanActions[chosenIndex]

    # Expectimax function
    def expectimaxFunction(self, currentDepth, gameState, agentIndex):
        if (self.depth == currentDepth or gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)
        minMoves = gameState.getLegalActions(agentIndex)
        resultStates = [gameState.generateSuccessor(agentIndex, action) for action in minMoves]

        if (agentIndex >= gameState.getNumAgents() - 1):
            score = [self.maximumFunction(currentDepth + 1, state, agentIndex) for state in resultStates]
        else:
            score = [self.expectimaxFunction(currentDepth, state, agentIndex + 1) for state in resultStates]
        return sum(score)/len(score)

    # Maximum function
    def maximumFunction(self, currentDepth, gameState, agentIndex):
        if (self.depth == currentDepth or gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)
        maxMoves = gameState.getLegalActions(0)
        resultStates = [gameState.generateSuccessor(0, action) for action in maxMoves]
        score = [self.expectimaxFunction(currentDepth, state, 1) for state in resultStates]
        return max(score)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()
# Abbreviation
better = betterEvaluationFunction
