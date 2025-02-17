# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qval = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qval[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state):
          return 0.0
        qval = float('-inf')
        for action in self.getLegalActions(state):
          qval = max(qval, self.getQValue(state, action))
        return qval

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions:
            return None
        qaction = None
        currentqval = float('-inf')
        for action in self.getLegalActions(state):
            if currentqval < self.getQValue(state, action):
                qaction = action
            currentqval = max(currentqval, self.getQValue(state, action))
        return qaction
         

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if not legalActions:
            return action
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        self.qval[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (reward + self.discount * self.getQValue(nextState, self.getPolicy(nextState)))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        qval = 0
        for feature in features:
            qval += self.weights[feature] * features[feature]
        return qval

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # features = self.featExtractor.getFeatures(state, action)
        # for feature in features:
        #   self.weights[feature] += self.alpha * (reward + self.discount * self.getQValue(nextState, self.getPolicy(nextState)) - self.getQValue(state, action)) * features[feature]
        features = self.featExtractor.getFeatures(state, action)
        diff = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        for feature in features:
          self.weights[feature] += self.alpha * diff * features[feature]


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass



class BetterExtractor(FeatureExtractor):
    "Your extractor entry goes here.  Add features for capsuleClassic."
    
    def getFeatures(self, state, action):
        features = SimpleExtractor().getFeatures(state, action)
        # Add more features here
        "*** YOUR CODE HERE ***"
        walls = state.getWalls()
        ghosts = state.getGhostStates()
        ghostPos = state.getGhostPositions()
        capsules = state.getCapsules()
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        
        # 地图归一常数
        grid_area = walls.width * walls.height
        open_area_ratio = getOpenAreaRatio(walls) 
        
        # 最近capsule
        closest_dist_to_capsule = closestPos((next_x, next_y), capsules, walls)
        closest_dist_to_ghost = closestPos((next_x, next_y), ghostPos, walls)
        
        # 受惊和非受惊幽灵
        ghostsPosInt = [(int(g.getPosition()[0] + 0.5), int(g.getPosition()[1] + 0.5)) for g in ghosts]
        active_ghosts = [g for g in ghosts if g.scaredTimer == 0]
        scared_ghosts = [g for g in ghosts if g.scaredTimer > 0]
        scare_time = min(g.scaredTimer for g in ghosts)
        
        # 危险走廊
        corridor_danger = dangerCorridor((next_x, next_y), ghostsPosInt, walls)
        dangerous_corridor = isDangerousCorridor((next_x, next_y), ghosts, walls, capsules)
        
        # 可以走的地方 
        legal_neighbors = Actions.getLegalNeighbors((next_x, next_y), walls)
        
        # features["x+y"] = (x + y) / open_area_ratio
        features["x+y"] = (x + y) / grid_area
        features["closest-capsule"] = float(closest_dist_to_capsule) / grid_area if closest_dist_to_capsule is not None else 0
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) for g in active_ghosts) / len(ghosts)
        # features["ghosthunter"] = 1
        # for g in ghosts:
        #     if (g in active_ghosts and closest_dist_to_ghost != closestPos((next_x, next_y), g.getPosition(), walls)):
        #         if (features["ghosthunter"] != 0):
        #             features["ghosthunter"] = 1 / (len(scared_ghosts) + 5)
        #     elif (g in active_ghosts and closest_dist_to_ghost == closestPos((next_x, next_y), g.getPosition(), walls)):
        #         features["ghosthunter"] = 0
        #     elif (g in scared_ghosts and closest_dist_to_ghost != closestPos((next_x, next_y), g.getPosition(), walls)):
        #         if (features["ghosthunter"] != 0):
        #             features["ghosthunter"] = 1 / (len(scared_ghosts) + 5)
        #             features["closest-capsule"] = float(closest_dist_to_capsule) / grid_area if closest_dist_to_capsule is not None else 0
        #     elif (g in scared_ghosts and closest_dist_to_ghost == closestPos((next_x, next_y), g.getPosition(), walls)):
        #         if (closest_dist_to_capsule is not None):
        #             if (closest_dist_to_ghost is not None):
        #                 if (closest_dist_to_ghost <= closest_dist_to_capsule + 2):
        #                     features["ghosthunter"] = 1
        #                     features["closest-capsule"] = 0
        #                 else:
        #                     features["ghosthunter"] = 0
        #                     features["closest-capsule"] = float(closest_dist_to_capsule) / grid_area if closest_dist_to_capsule is not None else 0
        #             else:
        #                 features["ghosthunter"] = 0
        #                 features["closest-capsule"] = float(closest_dist_to_capsule) / grid_area if closest_dist_to_capsule is not None else 0
        #         else:
        #             features["ghosthunter"] = 1
        #             features["closest-capsule"] = 0
        #     else:
        #         if (features["ghosthunter"] != 0):
        #             features["ghosthunter"] = 1 / (len(scared_ghosts) + 5)
        #             features["closest-capsule"] = float(closest_dist_to_capsule) / grid_area if closest_dist_to_capsule is not None else 0
        # i = 0
        # for g in ghosts:
        #     if(closestPos((next_x, next_y), ghost_pos[i], walls) is not None):
        #         if(closestPos((next_x, next_y), ghost_pos[i], walls) < 5 and closestPos((next_x, next_y), ghost_pos[i], walls) > 2):
        #             if(ghosts[i].scaredTimer > 5):  
        #                 features["scared"] = 1
        #                 pass
        #             else:
        #                 features["closest-capsule"] = 0
        #     i += 1
        # if len(active_ghosts) > 0:
        #     features["#-of-ghosts-1-step-away"] *= 1.5
        # if closest_dist_to_capsule is not None and closest_dist_to_capsule < 5:
        #     features["closest-capsule"] *= 2 
        
        
        features["scare num"] = 1 / (len(scared_ghosts) + 5)
        features["dangerous-corridor"] = 1 if dangerous_corridor else 0
        # features["danger-corridor"] = corridor_danger / len(ghosts) if len(ghosts) > 0 else 0
        features["deadend"] = 1 / (safeArea((next_x, next_y), ghostsPosInt, walls) + 1)
        # if (next_x, next_y) in ghostPos and scare_time > 20:
        #   features["ghost_hunting"] = 1
        if len(legal_neighbors) <= 2 and (next_x, next_y) in ghostsPosInt:  # Narrow corridor or dead-end
            features["dead-end-penalty"] = 1
        else:
            features["dead-end-penalty"] = 0


        features.divideAll(max(sum(abs(value) for value in features.values()), 1))
        
        return features

from collections import deque

def closestPos(pos, targets, walls, max_depth=None):
    if not targets:
        return None

    fringe = deque([(pos[0], pos[1], 0)])
    expanded = set()

    while fringe:
        pos_x, pos_y, dist = fringe.popleft()
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))

        if (pos_x, pos_y) in targets:
            return dist

        if max_depth and dist >= max_depth:
            continue

        for nbr_x, nbr_y in Actions.getLegalNeighbors((pos_x, pos_y), walls):
            fringe.append((nbr_x, nbr_y, dist + 1))

    return None

def safeArea(pos, ghosts, walls, max_area=100):
    fringe = deque([(pos[0], pos[1])])
    expanded = set()
    area = 0

    while fringe:
        pos_x, pos_y = fringe.popleft()
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        area += 1

        if area > max_area:
            return max_area

        if (pos_x, pos_y) in ghosts:
            continue

        for nbr_x, nbr_y in Actions.getLegalNeighbors((pos_x, pos_y), walls):
            fringe.append((nbr_x, nbr_y))

    return area

def dangerCorridor(pos, ghosts, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    corridor_depth = 0
    ghost_in_corridor = 0
    corridor_width = 0

    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))

        legal_neighbors = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        if len(legal_neighbors) <= 2:
            corridor_width += 1
            corridor_depth = max(corridor_depth, dist)

        if (pos_x, pos_y) in ghosts:
            ghost_in_corridor += 1

        for nbr_x, nbr_y in legal_neighbors:
            fringe.append((nbr_x, nbr_y, dist + 1))

    return ghost_in_corridor * corridor_depth / (corridor_width + 1)



def isDangerousCorridor(pos, ghosts, walls, capsules):
    fringe = [pos]
    expanded = set()
    corridor_width = 0

    while fringe:
        current = fringe.pop()
        if current in expanded:
            continue
        expanded.add(current)
        
        nbrs = Actions.getLegalNeighbors(current, walls)
        if len(nbrs) <= 2: 
            corridor_width += 1
            for nbr in nbrs:
                fringe.append(nbr)
    
    contains_dangerous_ghost = any(
        g.getPosition() in expanded and g.scaredTimer == 0 for g in ghosts
    )
    
    if len(capsules) > 0 and contains_dangerous_ghost:
        return True
    
    return False


def getOpenAreaRatio(walls):
    total_cells = walls.width * walls.height
    wall_cells = sum(1 for x in range(walls.width) for y in range(walls.height) if walls[x][y])
    open_cells = total_cells - wall_cells
    return open_cells / total_cells