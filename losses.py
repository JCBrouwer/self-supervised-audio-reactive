import torch

from models.slowfast import SlowFastExtractor
from models.vggish import VggishExtractor

slowfast = SlowFastExtractor()
output = slowfast(torch.randn((4, 32, 3, 256, 256)))
for o in output:
    print(o.shape)

vggish = VggishExtractor()
output = vggish(torch.rand((4, 44100, 2)) * 2 - 1)
for o in output:
    print(o.shape)
