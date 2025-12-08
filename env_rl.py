from cube2x2 import Cube2x2
import numpy as np

class CubeEnvRL:
    def __init__(self):
        self.cube=Cube2x2()

    def reset(self,depth=5):
        self.cube=Cube2x2()
        self.cube.scramble(depth)
        return self.cube.to_onehot()

    def step(self,action):
        move=Cube2x2.MOVES[action]
        self.cube.move(move)
        next_state=self.cube.to_onehot()
        reward=1.0 if self.cube.is_solved() else -0.1
        done=self.cube.is_solved()
        return next_state, reward, done
