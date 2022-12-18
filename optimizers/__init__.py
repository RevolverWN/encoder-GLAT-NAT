import os
import importlib
from collections import defaultdict

registry = defaultdict(dict)

def setup_optimizer(optimizer_name):
    optimizer = registry[optimizer_name]
    return optimizer


def register_optimizer(optimizer_name):
    def register_optimizer_cls(cls):
        registry[optimizer_name]["cls"] = cls
        registry[optimizer_name]["default_config_dict"] = cls.config
        return cls

    return register_optimizer_cls


dirname, base_name = os.path.split(__file__)
for filename in os.listdir(dirname):
    if filename != base_name and filename.endswith(".py"):
        importlib.import_module("optimizers." + filename[: filename.rfind(".py")])