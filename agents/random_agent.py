"""
Used as a baseline, this agent behaves completely randomly
"""
from agents.base_agent import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
    def __init__(self, obs_dim: int, action_dim: int , config: dict):
        super().__init__(obs_dim, action_dim, config)
    
    def act(self, observation: np.ndarray, deterministic=False) -> tuple[int, float, float]:
        return np.random.randint(self.action_dim), None, None
    
    def update(self, *args, **kwargs):
        pass

    def store_experience(self, *args, **kwargs):
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass