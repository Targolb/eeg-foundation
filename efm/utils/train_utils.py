import os, random, numpy as np, torch


def seed_all(seed=42):
    random.seed(seed);
    np.random.seed(seed)
    torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True


def save_ckpt(path, model):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict()}, path)


def load_yaml(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)
