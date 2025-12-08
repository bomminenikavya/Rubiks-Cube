import numpy as np
import random

class Cube2x2:
    def __init__(self):
        # 8 corners, each with 0,1,2 orientation
        self.perm = np.arange(8)
        self.orient = np.zeros(8, int)

    def clone(self):
        c = Cube2x2()
        c.perm = self.perm.copy()
        c.orient = self.orient.copy()
        return c

    def is_solved(self):
        return np.all(self.perm == np.arange(8)) and np.all(self.orient == 0)

    # ================================
    # Corner permutation table
    # ================================
    MOVE_PERM = {
        'U':  [1, 2, 3, 0, 4, 5, 6, 7],
        "U'": [3, 0, 1, 2, 4, 5, 6, 7],

        'R':  [0, 5, 2, 3, 4, 6, 7, 1],
        "R'": [0, 7, 2, 3, 4, 1, 5, 6],

        'F':  [3, 1, 2, 7, 0, 5, 6, 4],
        "F'": [4, 1, 2, 0, 7, 5, 6, 3],

        'L':  [0, 1, 6, 2, 4, 5, 3, 7],
        "L'": [0, 1, 3, 7, 4, 5, 2, 6],

        'D':  [0, 1, 2, 3, 5, 6, 7, 4],
        "D'": [0, 1, 2, 3, 7, 4, 5, 6],

        'B':  [0, 2, 6, 3, 4, 1, 5, 7],
        "B'": [0, 5, 1, 3, 4, 2, 6, 7]
    }

    # ================================
    # Corner orientation (0 for 2×2 basic version)
    # Can be extended later if needed
    # ================================
    MOVE_ORIENT = {m: [0] * 8 for m in MOVE_PERM}

    # List of all moves
    MOVES = list(MOVE_PERM.keys())

    # ================================
    # Apply a move
    # ================================
    def move(self, m):
        mp = self.MOVE_PERM[m]
        mo = self.MOVE_ORIENT[m]

        newp = np.zeros(8, int)
        newo = np.zeros(8, int)

        for i in range(8):
            old = mp[i]
            newp[i] = self.perm[old]
            newo[i] = (self.orient[old] + mo[i]) % 3

        self.perm = newp
        self.orient = newo

    # ================================
    # Scramble the cube with random moves
    # ================================
    def scramble(self, d=10):
        for _ in range(d):
            self.move(random.choice(self.MOVES))

    # ================================
    # Convert cube state to one-hot encoding
    # ================================
    def to_onehot(self):
        vec = []

        # Corner positions (8 corners × 8 possible locations)
        for i in range(8):
            oh = [0] * 8
            oh[self.perm[i]] = 1
            vec += oh

        # Corner orientation (8 corners × 3 orientation states)
        for i in range(8):
            oh = [0] * 3
            oh[self.orient[i]] = 1
            vec += oh

        return np.array(vec, np.float32)
