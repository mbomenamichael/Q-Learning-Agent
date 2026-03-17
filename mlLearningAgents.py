# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util

class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm
    """
    
    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """
        self.pacman_pos = state.getPacmanPosition()
        self.ghost_positions = state.getGhostPositions()
        self.food = state.getFood()
        self.score = state.getScore()

class QLearnAgent(Agent):
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining=10):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.numTraining = numTraining
        self.weights = {'num_ghosts_nearby': 0.0, 'eats_food': 0.0, 'closest_food': 0.0, 'bias': 0.0}
        
     def getFeatures(self, state, action):
        features = {}
        nextState = self.getSuccessor(state, action)
        pacman_pos = nextState.getPacmanPosition()
        features['bias'] = 1.0
        features['num_ghosts_nearby'] = sum(1 for ghost in nextState.getGhostPositions() if util.manhattanDistance(pacman_pos, ghost) == 1)
        features['eats_food'] = 1.0 if nextState.getFood()[pacman_pos[0]][pacman_pos[1]] else 0.0
        features['closest_food'] = min([util.manhattanDistance(pacman_pos, food) for food in nextState.getFood().asList()] or [0])
        return features
    
    @staticmethod
    def computeReward(startState: GameState, endState: GameState) -> float:
        if endState.isWin():
            return 100  # Reward for winning
        if endState.isLose():
            return -100  # Penalty for losing
        return endState.getScore() - startState.getScore()
    
    def getQValue(self, state, action):
        features = self.getFeatures(state, action)
        return sum(self.weights[f] * value for f, value in features.items())
    
    def maxQValue(self, state: GameStateFeatures) -> float:
        legal_actions = [a for a in Directions.__dict__.values() if isinstance(a, str)]
        if not legal_actions:
            return 0.0
        return max(self.getQValue(state, action) for action in legal_actions)
    
    def learn(self, state: GameStateFeatures, action: Directions, reward: float, nextState: GameStateFeatures):
        q_value = self.getQValue(state, action)
        max_q_next = self.maxQValue(nextState)
        new_q_value = (1 - self.alpha) * q_value + self.alpha * (reward + self.gamma * max_q_next)
        self.q_values[state.pacman_pos][action] = new_q_value
    
    def updateCount(self, state: GameStateFeatures, action: Directions):
        self.state_action_counts[state.pacman_pos][action] += 1
    
     def update(self, state, action, reward, nextState):
        correction = (reward + self.gamma * max(self.getQValue(nextState, a) for a in nextState.getLegalActions()) - self.getQValue(state, action))
        features = self.getFeatures(state, action)
        for f in features:
            self.weights[f] += self.alpha * correction * features[f]
    
    def explorationFn(self, utility: float, counts: int) -> float:
        if counts < 1:
            return float('inf')  # Encourages unexplored actions
        return utility + (1.0 / counts)
    
    def getAction(self, state: GameState) -> Directions:
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        
        stateFeatures = GameStateFeatures(state)
        if util.flipCoin(self.epsilon):
            return random.choice(legal)
        
        best_action = max(legal, key=lambda action: self.explorationFn(self.getQValue(stateFeatures, action), self.getCount(stateFeatures, action)))
        return best_action
    
    def final(self, state: GameState):
        self.episodesSoFar += 1
        if self.episodesSoFar == self.numTraining:
            print('Training Done (turning off epsilon and alpha)')
            self.alpha = 0
            self.epsilon = 0
