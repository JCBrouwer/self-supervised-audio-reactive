import json
import urllib
from functools import partial
from pathlib import Path

import torch
import wget
from munch import Munch
from torch import nn

from slowfast.models.build import MODEL_REGISTRY
from slowfast.utils.checkpoint import load_checkpoint
from slowfast.utils.parser import load_config as load_slowfast_config


def load_slowfast_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg


def ensure_parents(path):
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(str(path), "r") as f:
        data = json.load(f)
    return data


def dump_json(data, path, indent=4):
    ensure_parents(path)
    with open(str(path), "w") as f:
        json.dump(data, f, indent=indent)
    return


def read_url(url):
    with urllib.request.urlopen(url) as f:
        lines = []
        for line in f:
            line = line.decode("utf-8")
            lines.append(line)
    return lines


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


def set_cache_dir(cache_dir):
    cache_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    return cache_dir


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


def load_model_options(cache_dir):
    cache_path = cache_dir / "choices.json"
    return load_with_cache(cache_path, get_model_zoo)


def get_model_zoo():
    model_zoo_url = "https://raw.githubusercontent.com/facebookresearch/SlowFast/master/MODEL_ZOO.md"
    model_zoo = read_url(model_zoo_url)
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


def preprocess(visual, audio, cfg):
    frames, fps = visual
    frames = _preprocess(cfg, frames)
    return {"data": frames, "fps": fps}


"""
model_tags = {
    'Kinetics/c2/SLOWFAST_8x8_R50':
        {
            'name': 'SLOWFAST_8x8_R50',
            'dataset': 'kinetics-400',
        }
}
"""


class SlowFast(nn.Module):
    args = {"slowfast_config": "Kinetics/c2/SLOWFAST_8x8_R50"}
    model_tag = {
        "name": "SLOWFAST_8x8_R50",
        "dataset": "kinetics-400",
    }
    output_dims = 2304

    def __init__(self, args):
        super().__init__()

        # self.model_tag = self.model_tags[args.slowfast_config]
        self.cache_dir = set_cache_dir(args.data.cache_dir)
        self.model_choices = load_model_options(self.cache_dir)
        self.model, self.cfg = load_model(
            args.slowfast_config, self.model_choices, self.cache_dir, args.computation.device
        )

    @classmethod
    def download(cls, args):
        cache_dir = set_cache_dir(args.data.cache_dir)
        model_choices = load_model_options(cache_dir)
        model, cfg = load_model(args.slowfast_config, model_choices, cache_dir, args.computation.device)
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
