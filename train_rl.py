import torch, random
from env_rl import CubeEnvRL
from dqn_model import DQN
from replay_buffer import ReplayBuffer
import torch.nn.functional as F

device = "cpu"
env = CubeEnvRL()
net = DQN().to(device)
targ = DQN().to(device)
targ.load_state_dict(net.state_dict())
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
buf = ReplayBuffer(50000)

episodes = 2000
epsilon = 1.0
eps_min = 0.1
eps_decay = 0.995
gamma = 0.99

for ep in range(episodes):
    depth = min(5 + ep // 200, 10)
    state = env.reset(depth)

    # FIXED dtype error
    state = torch.tensor(state, dtype=torch.float32).to(device)

    done = False
    steps = 0

    while not done and steps < 50:
        steps += 1

        # ε-greedy
        if random.random() < epsilon:
            action = random.randint(0, 11)
        else:
            with torch.no_grad():
                action = torch.argmax(net(state)).item()

        next_state, rew, done = env.step(action)

        # FIXED dtype error
        next_state_t = torch.tensor(next_state, dtype=torch.float32).to(device)

        buf.add((state, action, rew, next_state_t, done))
        state = next_state_t

        batch = buf.sample(64)

        if len(batch) > 10:
            s, a, r, ns, d = zip(*batch)

            # FIXED dtype errors
            s = torch.stack(s).to(device)
            ns = torch.stack(ns).to(device)
            a = torch.tensor(a, dtype=torch.long).to(device)
            r = torch.tensor(r, dtype=torch.float32).to(device)
            d = torch.tensor(d, dtype=torch.float32).to(device)

            q = net(s).gather(1, a.unsqueeze(1)).squeeze()

            with torch.no_grad():
                max_next = targ(ns).max(1)[0]
                target = r + gamma * max_next * (1 - d)

            loss = F.mse_loss(q, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

    if epsilon > eps_min:
        epsilon *= eps_decay

    if ep % 100 == 0:
        targ.load_state_dict(net.state_dict())
        print("Episode", ep, "completed")

# Save DQN model safely for GitHub
torch.save(net.state_dict(), "dqn.bin")
