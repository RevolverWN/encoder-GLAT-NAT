import importlib
import os

registry = {}


def register_hook(register_name):
    def register_hook_cls(cls):
        registry[register_name] = cls

    return register_hook_cls


dir_name, base_name = os.path.split(__file__)
for filename in os.listdir(dir_name):
    if filename.endswith(".py") and filename != base_name:
        importlib.import_module("hooks." + filename[:filename.rfind(".py")])
