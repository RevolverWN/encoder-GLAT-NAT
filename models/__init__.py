import os
import importlib
from collections import defaultdict
from typing import Dict

registry: Dict[str, Dict] = defaultdict(dict)

def setup_model(model_name):
    model = registry[model_name]
    return model


def register_model(model_name):
    def register_model_cls(cls):
        registry[model_name]["cls"] = cls
        registry[model_name]["default_config_dict"] = cls.config
        return cls

    return register_model_cls


dirname, base_name = os.path.split(__file__)
for filename in os.listdir(dirname):
    if filename != base_name and filename.endswith(".py"):
        importlib.import_module("models." + filename[: filename.rfind(".py")])



