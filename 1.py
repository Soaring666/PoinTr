from utils.config import cfg_from_yaml_file
from pathlib import Path
import torch

step = 3
a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
print("a:", a, "b:", b)