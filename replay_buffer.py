import random

class ReplayBuffer:
    def __init__(self,cap=50000):
        self.cap=cap
        self.buf=[]
    def add(self,exp):
        if len(self.buf)>=self.cap:
            self.buf.pop(0)
        self.buf.append(exp)
    def sample(self,batch):
        return random.sample(self.buf,min(batch,len(self.buf)))
