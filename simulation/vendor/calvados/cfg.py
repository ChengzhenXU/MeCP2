"""CALVADOS configuration and component defaults."""
from importlib import resources
import yaml


class Config:
    def __init__(self,**params):
        self.params = params
        self.pkg_base = resources.files('calvados')
        self.config = self.load_default_config(self.pkg_base)
        self.load_config()

    def load_config(self):
        for key, val in self.params.items():
            self.config[key] = val

    @staticmethod
    def load_default_config(pkg_base):
        """ Load default config. """
        with open(f'{pkg_base}/data/default_config.yaml','r') as stream:
            default_config = yaml.safe_load(stream)
        return default_config


class Components:
    def __init__(self,**defaults):
        self.pkg_base = resources.files('calvados')
        self.components = {}

        self.components['defaults'] = self.load_default_component(self.pkg_base,defaults)
        self.components['system'] = {}

    def reset_components(self,**kwargs):
        self.components['system'] = {}

    def add(self,**kwargs):
        nm = kwargs['name']
        del kwargs['name']
        self.components['system'][nm] = kwargs

    def write(self,path,name='components.yaml'):
        """ Write component file. """
        self.name = name
        with open(f'{path}/{name}','w') as stream:
            yaml.dump(self.components,stream,sort_keys=False)

    @staticmethod
    def load_default_component(pkg_base,defaults):
        """ Load default config. """
        # package defaults
        with open(f'{pkg_base}/data/default_component.yaml','r') as stream:
            default_component = yaml.safe_load(stream)
        # manual defaults
        for key, val in defaults.items():
            default_component[key] = val
        return default_component
