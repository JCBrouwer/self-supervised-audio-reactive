"""
From https://github.com/sangho-vision/acav100m

MIT License

Copyright (c) 2021 Sangho Lee, Jiwan Chung, Youngjae Yu, Gunhee Kim, Thomas Breuel, Gal Chechik, Yale Song

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import json
import urllib
from argparse import Namespace
from functools import partial
from pathlib import Path

import torch
import wget
from munch import Munch
from torch import nn

from slowfast.datasets.utils import pack_pathway_output
from slowfast.datasets.utils import spatial_sampling as _spatial_sampling
from slowfast.datasets.utils import tensor_normalize
from slowfast.models.build import MODEL_REGISTRY
from slowfast.utils.checkpoint import load_checkpoint
from slowfast.utils.parser import load_config as load_slowfast_config


class SlowFastExtractor(nn.Module):
    def __init__(self, fps=24):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fps = fps
        self.slowfast = LayerSlowFast(
            Namespace(
                data=Namespace(cache_dir=Path("./cache")),
                computation=Namespace(device=self.device),
                slowfast_config="Kinetics/c2/SLOWFAST_8x8_R50",
                num_layers=5,
            )
        )
        self.preprocess = self.slowfast.get_preprocessor()

    def forward(self, x):
        """
        [B T C H W] Video Tensor -> [[B, F1], [B, F2], ..., [B, F5]] Feature Tensors
        """
        x = x.permute(0, 1, 3, 4, 2)  # slowfast expects channels last
        slow, fast = [], []
        for sample in x:
            sample = self.preprocess((sample, self.fps), None)
            fast.append(sample["data"][0])
            slow.append(sample["data"][1])
        slow, fast = torch.stack(slow).to(self.device), torch.stack(fast).to(self.device)
        output = self.slowfast([fast, slow])
        return output


class SlowFast(nn.Module):
    args = {"slowfast_config": "Kinetics/c2/SLOWFAST_8x8_R50"}
    model_tag = {
        "name": "SLOWFAST_8x8_R50",
        "dataset": "kinetics-400",
    }
    output_dims = 2304

    def __init__(self, args):
        super().__init__()
        self.cache_dir = args.data.cache_dir
        self.model_choices = load_with_cache(self.cache_dir / "choices.json", get_model_zoo)
        self.model, self.cfg = load_model(
            args.slowfast_config, self.model_choices, self.cache_dir, args.computation.device
        )

    @classmethod
    def download(cls, args):
        model_choices = load_with_cache(args.data.cache_dir / "choices.json", get_model_zoo)
        model, cfg = load_model(args.slowfast_config, model_choices, args.data.cache_dir, args.computation.device)
        return model

    def get_preprocessor(self):
        return partial(preprocess, cfg=self.cfg)

    def _forward(self, x):
        model = self.model
        x = model.s1(x)
        x = model.s1_fuse(x)
        x = model.s2(x)
        x = model.s2_fuse(x)
        for pathway in range(model.num_pathways):
            pool = getattr(model, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = model.s3(x)
        x = model.s3_fuse(x)
        x = model.s4(x)
        x = model.s4_fuse(x)
        x = model.s5(x)

        head = self.model.head
        assert len(x) == head.num_pathways, "Input tensor does not contain {} pathway".format(head.num_pathways)
        pool_out = []
        for pathway in range(head.num_pathways):
            m = getattr(head, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(x[pathway]))
        x = torch.cat(pool_out, 1)
        # (B, C, T, H, W) -> (B, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        return x

    def forward(self, data):
        x = data
        # BTHWC: BatchSize, NumFrames, Height, Width, Channels
        x = self._forward(x)
        x = x.mean([1, 2, 3])
        # BC
        return x


class LayerSlowFast(SlowFast):
    args = {"slowfast_config": "Kinetics/c2/SLOWFAST_8x8_R50", "num_layers": 5}
    output_dims = [88, 352, 704, 1408, 2304]

    def __init__(self, args):
        super().__init__(args)
        self.num_layers = args.num_layers

    def _forward(self, x):
        model = self.model
        xs = []
        x = model.s1(x)
        x = model.s1_fuse(x)
        xs.append([v.clone().detach() for v in x])
        x = model.s2(x)
        x = model.s2_fuse(x)
        for pathway in range(model.num_pathways):
            pool = getattr(model, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        xs.append([v.clone().detach() for v in x])
        x = model.s3(x)
        x = model.s3_fuse(x)
        xs.append([v.clone().detach() for v in x])
        x = model.s4(x)
        x = model.s4_fuse(x)
        xs.append([v.clone().detach() for v in x])
        x = model.s5(x)
        xs.append([v.clone().detach() for v in x])

        head = self.model.head
        assert len(x) == head.num_pathways, "Input tensor does not contain {} pathway".format(head.num_pathways)

        def get_pool(x):
            pool_out = []
            for pathway in range(head.num_pathways):
                m = getattr(head, "pathway{}_avgpool".format(pathway))
                pool_out.append(m(x[pathway]))
            x = torch.cat(pool_out, 1)
            # (B, C, T, H, W) -> (B, T, H, W, C).
            x = x.permute((0, 2, 3, 4, 1))
            return x

        xs = [get_pool(x) for x in xs]
        return xs

    def forward(self, data, no_grad=True):
        x = data
        # BTHWC: BatchSize, NumFrames, Height, Width, Channels
        if no_grad:
            for i, _ in enumerate(x):
                x[i].requires_grad_(False)
        xs = self._forward(x)
        xs = [x.mean([1, 2, 3]) for x in xs]
        # BC
        return xs


def preprocess(visual, audio, cfg):
    frames, fps = visual
    frames = _preprocess(cfg, frames)
    return {"data": frames, "fps": fps}


def _preprocess(cfg, x):
    x = tensor_normalize(x, cfg.DATA.MEAN, cfg.DATA.STD)
    # T H W C -> C T H W.
    x = x.permute(3, 0, 1, 2)
    x = spatial_sampling(cfg, x)
    x = pack_pathway_output(cfg, x)
    if isinstance(x, list):
        if any(v is None for v in x):
            return None
    return x


def spatial_sampling(cfg, frames):
    min_scale, max_scale, crop_size = [cfg.DATA.TEST_CROP_SIZE] * 3

    frames = _spatial_sampling(
        frames,
        spatial_idx=0,
        min_scale=min_scale,
        max_scale=max_scale,
        crop_size=crop_size,
        random_horizontal_flip=cfg.DATA.RANDOM_FLIP,
        inverse_uniform_sampling=cfg.DATA.INV_UNIFORM_SAMPLE,
    )

    return frames


def load_model(config, choices, cache_dir, device="cuda"):
    assert config in choices, f"no SlowFast backbone named {config}"
    config_name = f"{config}.yaml".replace("/", "_")
    config_path = cache_dir / config_name
    if not config_path.is_file():
        slowfast_url = "https://raw.githubusercontent.com/facebookresearch/SlowFast/master"
        config_url = f"{slowfast_url}/configs/{config}.yaml"
        download_file(config_url, config_path)
    args = build_slowfast_args(config_path)
    cfg = load_slowfast_config(args)
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)
    ckpt_path = load_weights(config, choices, cache_dir)
    convert_from_caffe2 = cfg.TEST.CHECKPOINT_TYPE == "caffe2"
    if config == "Kinetics/c2/SLOWFAST_8x8_R50":
        # The checkpoint files use caffe2 without telling us so
        convert_from_caffe2 = True
    epoch = load_checkpoint(str(ckpt_path), model, data_parallel=False, convert_from_caffe2=convert_from_caffe2)
    if not convert_from_caffe2:
        assert epoch > 0, "SlowFast ckpt not loaded!"
    model = model.to(device)
    model.eval()
    return model, cfg


def build_slowfast_args(config_path):
    args = {
        "shard_id": 0,
        "num_shards": 1,
        "init_method": "tcp://localhost:9999",
        "cfg_file": config_path,
        "opts": None,
    }
    return Munch(args)


def load_weights(config, choices, cache_dir):
    cache_path = cache_dir / f"{config.replace('/','_')}.pkl"
    if not cache_path.is_file():
        url = choices[config]
        download_file(url=url, out=cache_path)
    return cache_path


def get_model_zoo():
    model_zoo_url = "https://raw.githubusercontent.com/facebookresearch/SlowFast/master/MODEL_ZOO.md"
    with urllib.request.urlopen(model_zoo_url) as f:
        model_zoo = [line.decode("utf-8") for line in f]
    data = parse_model_zoo(model_zoo)
    return data


def parse_model_zoo(model_zoo):
    reading_models = 0
    data = {}
    ckpt_num = -1
    config_num = -1
    title = ""

    def split_line(line):
        line = line.split("|")
        line = [element.strip() for element in line]
        line = [element for element in line if len(element) > 0]
        return line

    for line in model_zoo:
        line = line.strip()
        if reading_models == 2:
            if len(line) == 0:
                reading_models = 0
            else:
                line = split_line(line)
                if config_num < 0:
                    model_ckpt = line[ckpt_num]
                    model_ckpt = model_ckpt[model_ckpt.find("https://") : -1]
                    model_config = model_ckpt.split("/")[-1].split(".")[0]
                    if title is not None:
                        model_config = f"{title}/c2/{model_config}"
                    else:
                        model_config = None
                else:
                    model_config = line[config_num]
                    model_ckpt = line[ckpt_num]
                    model_config = model_config.strip()
                    model_ckpt = model_ckpt[model_ckpt.find("https://") : -1]

                if model_ckpt and model_config:
                    data[model_config] = model_ckpt
        elif reading_models == 0:
            if line.startswith("| architecture |"):
                line = split_line(line)
                ckpt_num = [i for i, v in enumerate(line) if v == "model"]
                ckpt_num = ckpt_num[0] if len(ckpt_num) > 0 else -1
                config_num = [i for i, v in enumerate(line) if v == "config"]
                config_num = config_num[0] if len(config_num) > 0 else -1
                title = "AVA" if "AVA version" in line else None
                reading_models = 1
        elif reading_models == 1:
            reading_models = 2
    return data


def load_json(path):
    with open(str(path), "r") as f:
        data = json.load(f)
    return data


def dump_json(data, path, indent=4):
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as f:
        json.dump(data, f, indent=indent)
    return


def download_file(url, out=None):
    url = str(url)
    if out is not None:
        out = Path(out)
        if out.is_file():
            print(f"File already exists: {str(out)}")
        out = str(out)
    wget.download(url, out=out)


def load_with_cache(cache_path, get_file, load_file=load_json, dump_file=dump_json, dumped=False):
    cache_path = Path(cache_path)
    if cache_path.is_file():
        return load_file(cache_path)
    else:
        data = get_file()
        if not dumped:
            dump_file(data, cache_path)
        return data
