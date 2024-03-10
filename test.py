import torch

data = [
    [1,2],
    [3,4]
]

tensor = torch.tensor(data)

print(torch.einsum('ii->i', tensor))