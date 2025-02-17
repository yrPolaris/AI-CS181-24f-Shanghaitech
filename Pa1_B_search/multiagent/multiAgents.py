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

from typing import List, Tuple
from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]
    
    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood().asList()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newGhostPos : List[Tuple[int, int]] = []
        for ghost in newGhostStates:
            newGhostPos.append(ghost.getPosition())
            
        ifScared : bool = newScaredTimes[0] > 0
        if not ifScared and (newPos in newGhostPos):
            return -1.0
        if newPos in currentGameState.getFood().asList():
            return 1.0
        
        def DistfromNewPos(currentPos : Tuple[int, int]):
            return util.manhattanDistance(currentPos, newPos)
        
        closestFoodDist : List[Tuple[int, int]] = sorted(newFood, key=DistfromNewPos)
        closestGhostDist : List[Tuple[int, int]] = sorted(newGhostPos, key=DistfromNewPos)
        return (1 / DistfromNewPos(closestFoodDist[0])) - (1 / DistfromNewPos(closestGhostDist[0]))

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        ghostList: List[int] = list(range(1, gameState.getNumAgents()))
        
        def isEnd(state: GameState, depth: int) -> bool:
            if state.isWin() or state.isLose():
                return True
            if self.depth == depth:
                return True
            return False
        
        def minValue(state: GameState, depth: int, ghostIndex: int) -> int:
            if isEnd(state, depth):
                return self.evaluationFunction(state)
            
            value: int = 0xffffffff
            for action in state.getLegalActions(ghostIndex):
                if ghostIndex == ghostList[-1]:
                    value = min(value, maxValue(state.getNextState(ghostIndex, action), depth + 1))
                else:
                    value = min(value, minValue(state.getNextState(ghostIndex, action), depth, ghostIndex + 1))
            return value
        
        
        def maxValue(state: GameState, depth: int) -> int:
            if isEnd(state, depth):
                return self.evaluationFunction(state)

            value: int = -0xfffffffe
            for action in state.getLegalActions(0):
                value = max(value, minValue(state.getNextState(0, action), depth, 1))
            return value
        
        result: List[Tuple[str, int]] = [(action, minValue(gameState.getNextState(0, action), 0, 1)) for action in
                                      gameState.getLegalActions(0)]
        result.sort(key=lambda k: k[1], reverse=True)
        return result[0][0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        currentValue: float = -0xfffffffe
        alpha: float = -0xfffffffe
        beta: float = 0xffffffff
        nextAction: str = Directions.STOP
        
        legalActionList: List[str] = gameState.getLegalActions(0).copy()
        for action in legalActionList:
            nextState: GameState = gameState.getNextState(0, action)
            nextValue: float = self.getValue(nextState, 0, 1, alpha, beta)
            if nextValue > currentValue:
                currentValue = nextValue
                nextAction = action
            alpha = max(alpha, currentValue)
        return nextAction

    def maxValue(self, gameState: GameState,currentDepth: int, agentIndex: int,
                   alpha: float = -0xfffffffe, beta: float = 0xffffffff):
        value = -10000000000000
        legalActionList: List[str] = gameState.getLegalActions(agentIndex)
        for action in legalActionList:
            nextvalue = self.getValue(gameState.getNextState(agentIndex, action),currentDepth,
                                            agentIndex + 1, alpha, beta)
            value = max(value, nextvalue)
            if value > beta:
                return value
            alpha = max(alpha, value)
        return value

    def minValue(self, gameState: GameState,currentDepth: int, agentIndex: int,
                  alpha: float = -0xfffffffe, beta: float = 0xffffffff):
        value = 100000000000
        legalActionList = gameState.getLegalActions(agentIndex)
        for action in legalActionList:
            if agentIndex == gameState.getNumAgents() - 1:
                nextvalue = self.getValue(gameState.getNextState(agentIndex, action),currentDepth + 1,
                                             0, alpha, beta)
                value = min(value, nextvalue)
                if value < alpha:
                    return value
            else:
                nextvalue = self.getValue(gameState.getNextState(agentIndex, action),currentDepth,
                                             agentIndex + 1, alpha, beta)
                value = min(value, nextvalue)
                if value < alpha:
                    return value
            beta = min(beta, value)
        return value   
    
    def getValue(self, gameState: GameState,currentDepth: int = 0, agentIndex: int = 0,
                     alpha: float = -0xfffffffe, beta: float = 0xffffffff):
        maxAgentList: List[int] = [0]
        minAgentList: List[int] = list(range(1, gameState.getNumAgents()))
        
        if currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        if agentIndex in maxAgentList:
            return self.maxValue(gameState, currentDepth, agentIndex, alpha, beta)
        elif agentIndex in minAgentList:
            return self.minValue(gameState, currentDepth, agentIndex, alpha, beta) 


    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        maxValue: float = -0xfffffffe
        maxAction: str = Directions.STOP
        
        for action in gameState.getLegalActions(agentIndex=0):
            SuccessorState: GameState = gameState.getNextState(0, action)
            SuccessorValue: float = self.expValue(SuccessorState, 0, 1)
            if SuccessorValue > maxValue:
                maxValue = SuccessorValue
                maxAction = action
        return maxAction

    def maxValue(self, gameState: GameState, currentDepth: int):
        if currentDepth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        
        maxValue: float = -0xfffffffe
        for action in gameState.getLegalActions(agentIndex=0):
            SuccessorState: GameState = gameState.getNextState(action=action, agentIndex=0)
            SuccessorValue: float = self.expValue(SuccessorState, currentDepth=currentDepth, agentIndex=1)
            if SuccessorValue > maxValue:
                maxValue = SuccessorValue
        return maxValue
    
    def expValue(self, gameState: GameState, currentDepth: int, agentIndex: int):
        if currentDepth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        
        actionsNum: int = len(gameState.getLegalActions(agentIndex=agentIndex))
        totalValue: float = 0.0
        agentNum: int = gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex=agentIndex):
            SuccessorState: GameState = gameState.getNextState(agentIndex=agentIndex, action=action)
            if agentIndex == agentNum - 1:
                SuccessorValue = self.maxValue(SuccessorState, currentDepth=currentDepth + 1)
            else:
                SuccessorValue = self.expValue(SuccessorState, currentDepth=currentDepth, agentIndex=agentIndex + 1)
            totalValue += SuccessorValue
        return totalValue / actionsNum
            
            
def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    
    "*** YOUR CODE HERE ***"
    currentPos: Tuple[int, int] = currentGameState.getPacmanPosition()
    currentFoodGrid: List[Tuple[int, int]] = currentGameState.getFood().asList()
    currentGhosts: List = currentGameState.getGhostStates()
    score: float = currentGameState.getScore()
    
    foodValue: float = 10.0
    ghostValue: float = -10.0
    scaredghostValue: float = 100.0
    
    foodDistList: List[int] = [util.manhattanDistance(currentPos, foodPos) for foodPos in currentFoodGrid]
    if len(foodDistList) > 0:
        score += foodValue / min(foodDistList)
        
    for ghost in currentGhosts:
        PacmanGhostDist: float = util.manhattanDistance(currentPos, ghost.getPosition())
        if PacmanGhostDist > 0:
            if ghost.scaredTimer > 0:
                score += scaredghostValue / PacmanGhostDist
            else:
                score += ghostValue / PacmanGhostDist
        else:
            return -0xfffffffe
        
    return score




# Abbreviation
better = betterEvaluationFunction
