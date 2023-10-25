import importlib
import logging

MODEL_MODULES = ["models"]
ALL_MODULES = [("model", MODEL_MODULES)]

def import_all_modules_for_register(custom_module_paths=None):
    modules = []
    for base_dir, module in ALL_MODULES:
        for name in module:
            full_name = base_dir + "." + name
            modules.append(full_name)
        if isinstance(custom_module_paths, list):
            modules += custom_module_paths
        errors = []
        for module in modules:
            try:
            # importlib动态导入
                importlib.import_module(module)
            except ImportError as error:
                errors.append((module, error))

class Register:
    def __init__(self, registry_name):
        self.dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable")
        if key is None:
            key = value.__name__
        if key in self.dict:
            logging.warning("Key %s already in registry %s." % (key, self.__name__))
        self.dict[key] = value

    def register(self, target):
        def add(key, value):
            self[key] = value
            return value
        if callable(target):
            return add(None, target)
        return lambda x: add(target, x)

    def __getitem__(self, key):
        return self.dict[key]

    def __contains__(self, key):
        return key in self.dict

    def keys(self):
        return self.dict.keys()

class Registers:
    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")

    model = Register('model')
