from abc import ABC, abstractmethod
import numpy as np

#The need for an abstract class is due to the fact that the training file is agnostic of the type of agent which is 
#currently running, otherwise a different training file would have to be written for each different agent implementation

# Base class for all agents, they inherit from this (abstract) class and implement the required methods
class BaseAgent(ABC):
    def __init__(self, obs_dim: int, action_dim: int , config: dict):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
    
    #Method to be implemented by all agents, it takes an observation and returns an action
    @abstractmethod
    def act(self, observation: np.ndarray, deterministic = False) -> tuple[int, float, float]:
        pass

    #Generic signature since different agents may require different arguments
    @abstractmethod
    def update(self, *args, **kwargs):
        pass
    
    #Save the model parameters locally
    @abstractmethod
    def save(self, path: str):
        pass
    
    #Load the model parameters from a local file
    @abstractmethod
    def load(self, path: str):
        pass

    #Store the experience in the buffers, training phase
    @abstractmethod
    def store_experience(self, *args, **kwargs):
        pass