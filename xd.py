import numpy as np
import torch

device = torch.device("mps")

conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3).to(device)

data = torch.tensor(np.random.uniform(size=(1, 10, 10, 1)), dtype=torch.float32).to(device)
optimizer = torch.optim.Adam(conv.parameters(), lr=0.1)
x = data.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
out = torch.sum(conv(x))
loss = torch.nn.MSELoss()(out, torch.zeros_like(out))
optimizer.zero_grad()
out.backward()
optimizer.step()
