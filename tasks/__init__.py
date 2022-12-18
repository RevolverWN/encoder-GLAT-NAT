import importlib
import os
from collections import defaultdict

registry = defaultdict(dict)


def register_task(task_name):
    def register_task_cls(cls):
        registry[task_name]["cls"] = cls
        registry[task_name]["default_config_dict"] = cls.config
        return cls

    return register_task_cls


dirname, base_name = os.path.split(__file__)
for filename in os.listdir(dirname):
    if filename != base_name and filename.endswith(".py"):
        importlib.import_module("tasks." + filename[: filename.rfind(".py")])