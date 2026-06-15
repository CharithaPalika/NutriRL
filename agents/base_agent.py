import torch
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(
        self,
        env,
        device="cpu",
        **kwargs,
    ):
        self.env = env
        self.device = device
        

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def generate_episode(self):
        pass

    @abstractmethod
    def infer_episode(self, memory):    
        pass
        
