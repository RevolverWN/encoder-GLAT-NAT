import os
import importlib
from collections import defaultdict

registry = defaultdict(dict)


def register_scheduler(scheduler_name):
    def register_scheduler_cls(cls):
        registry[scheduler_name]["cls"] = cls
        registry[scheduler_name]["default_config_dict"] = cls.config
        return cls

    return register_scheduler_cls


dirname, base_name = os.path.split(__file__)
for filename in os.listdir(dirname):
    if filename != base_name and filename.endswith(".py"):
        importlib.import_module("lr_schedulers." + filename[: filename.rfind(".py")])