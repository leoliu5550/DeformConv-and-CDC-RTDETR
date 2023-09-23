import os
import yaml 
import inspect
import importlib
__all__ = ['GLOBAL_CONFIG', 'register', 'create', 'load_config', 'merge_config', 'merge_dict']
GLOBAL_CONFIG = dict()
INCLUDE_KEY = '__include__'


def register(cls: type):
    '''
    Args:
        cls (type): Module class to be registered.
    '''
    if cls.__name__ in GLOBAL_CONFIG:
        raise ValueError('{} already registered'.format(cls.__name__))

    if inspect.isfunction(cls):
        GLOBAL_CONFIG[cls.__name__] = cls
    
    elif inspect.isclass(cls):
        GLOBAL_CONFIG[cls.__name__] = extract_schema(cls)

    else:
        raise ValueError(f'register {cls}')

    return cls 

def extract_schema(cls: type):
    '''
    Args:
        cls (type),
    Return:
        Dict, 
    '''
    argspec = inspect.getfullargspec(cls.__init__)
    arg_names = [arg for arg in argspec.args if arg != 'self']
    num_defualts = len(argspec.defaults) if argspec.defaults is not None else 0
    num_requires = len(arg_names) - num_defualts

    schame = dict()
    schame['_name'] = cls.__name__
    schame['_pymodule'] = importlib.import_module(cls.__module__)
    schame['_inject'] = getattr(cls, '__inject__', [])
    schame['_share'] = getattr(cls, '__share__', [])

    for i, name in enumerate(arg_names):
        if name in schame['_share']:
            assert i >= num_requires, 'share config must have default value.'
            value = argspec.defaults[i - num_requires]
        
        elif i >= num_requires:
            value = argspec.defaults[i - num_requires]

        else:
            value = None 

        schame[name] = value
        
    return schame



def create(type_or_name, **kwargs):
    '''
    '''
    assert type(type_or_name) in (type, str), 'create should be class or name.'

    name = type_or_name if isinstance(type_or_name, str) else type_or_name.__name__

    if name in GLOBAL_CONFIG:
        if hasattr(GLOBAL_CONFIG[name], '__dict__'):
            return GLOBAL_CONFIG[name]
    else:
        raise ValueError('The module {} is not registered'.format(name))

    cfg = GLOBAL_CONFIG[name]

    if isinstance(cfg, dict) and 'type' in cfg:
        _cfg: dict = GLOBAL_CONFIG[cfg['type']]
        _cfg.update(cfg) # update global cls default args 
        _cfg.update(kwargs) # TODO
        name = _cfg.pop('type')
        
        return create(name)