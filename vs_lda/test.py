import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

tensor_list = [
    torch.tensor([1.0, 1.0, 1.0]),
    torch.tensor([2.0, 2.0, 2.0]),
    torch.tensor([3.0, 3.0, 3.0]),
]

test = {
    "a": list(map(lambda x: x.tolist(), tensor_list)),
    "b": list(map(lambda x: x.tolist(), tensor_list)),
}

with open("./test.json", "w") as f:
    json.dump(test, f)
