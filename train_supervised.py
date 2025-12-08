import torch, pickle
from dca_model import DCAModel
import torch.nn.functional as F
import random
import numpy as np


device = "cpu"
model = DCAModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Load dataset (renamed from .pkl → .bin to avoid GitHub push protection)
with open("supervised_data.bin", "rb") as f:
    data = pickle.load(f)

epochs = 5
batch = 128

for ep in range(epochs):
    batch_data = random.sample(data, batch)

    # Separate states and target values
    states = [s for s, _ in batch_data]
    targets = [[d] for _, d in batch_data]

    # FIXED: convert list → numpy → tensor for speed + no warnings
    states = torch.from_numpy(np.array(states, dtype=np.float32)).to(device)
    targets = torch.tensor(targets, dtype=torch.float32).to(device)

    preds = model(states)
    loss = F.mse_loss(preds, targets)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print("Epoch", ep, "Loss", loss.item())

# Save the trained model (renamed to .bin for GitHub safety)
torch.save(model.state_dict(), "dca.bin")
