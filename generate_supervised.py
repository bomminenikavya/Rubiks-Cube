from cube2x2 import Cube2x2
import pickle, random

N=50000
depths=[1,2,3,4,5,6,7,8]
data=[]

for _ in range(N):
    c=Cube2x2()
    d=random.choice(depths)
    c.scramble(d)
    data.append((c.to_onehot(), d))

with open("supervised_data.pkl","wb") as f:
    pickle.dump(data,f)

print("Generated supervised_data.pkl with",N,"samples.")
