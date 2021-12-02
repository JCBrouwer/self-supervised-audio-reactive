import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.append("acav100m/feature_extraction/code")

from acav100m.feature_extraction.code.models.slowfast import LayerSlowFast
from acav100m.feature_extraction.code.models.vggish import LayerVggish

slowfast = LayerSlowFast(
    Namespace(
        data=Namespace(cache_dir=Path("./cache")),
        computation=Namespace(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        slowfast_config="Kinetics/c2/SLOWFAST_8x8_R50",
        num_layers=5,
    )
)
preprocess = slowfast.get_preprocessor()

slow, fast = [], []
for _ in range(1):
    sample = preprocess((torch.randn((32, 256, 256, 3)), 30), None)
    fast.append(sample["data"][0])
    slow.append(sample["data"][1])
slow, fast = torch.stack(slow).cuda(), torch.stack(fast).cuda()
output = slowfast([fast, slow])
for o in output:
    print(o.shape)

vggish = LayerVggish(
    Namespace(
        data=Namespace(cache_dir=Path("./cache")),
        postprocess=False,
        num_layers=5,
    )
).cuda()
preprocess = vggish.get_preprocessor()

batch = []
for _ in range(1):
    sample = preprocess(None, (torch.rand(44100, 2), 44100))["data"]
    batch.append(sample)
batch = torch.stack(batch).cuda()
output = vggish(batch)
for o in output:
    print(o.shape)
