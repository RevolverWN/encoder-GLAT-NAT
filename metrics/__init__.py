import importlib
import os

registry = {}


def register_metric(register_name):
    def register_metric_cls(cls):
        registry[register_name] = cls

    return register_metric_cls


dir_name, base_name = os.path.split(__file__)
for filename in os.listdir(dir_name):
    if filename.endswith(".py") and filename != base_name:
        importlib.import_module("metrics." + filename[:filename.rfind(".py")])

