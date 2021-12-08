import torch
from tqdm import tqdm

from models.patch_contrastive import CombinationsPatchContrastor, PatchSampler1d
from models.slowfast import SlowFastExtractor
from models.vggish import VggishExtractor

slowfast = SlowFastExtractor()
vggish = VggishExtractor()

sampler = PatchSampler1d(32, 64)
contrastor = CombinationsPatchContrastor(32)

optimizer = torch.optim.Adam(contrastor.params(), lr=1e-4)

for audio_features, video_features in tqdm(dataloader):
    optimizer.zero_grad()
    patches, _ = sampler(audio_features + audio_features)
    contrastor(patches).backward()
    optimizer.step()
