from trallie.providers.base import BaseProvider
from trallie.providers.errors import ProviderInitializationError

_PROVIDERS_REGISTRY = dict()
_PROVIDERS_REGISTRY_CLASS_NAMES = set()


def register_provider(name):
    def register_provider_cls(cls):
        if name in _PROVIDERS_REGISTRY:
            raise ValueError(
                f"Cannot register <{name}, {cls.__name__}>: duplicate "
                f"name '{name}'."
            )
        if cls.__name__ in _PROVIDERS_REGISTRY_CLASS_NAMES:
            raise ValueError(
                f"Cannot register <{name}, {cls.__name__}>: duplicate "
                f"class name '{cls.__name__}'."
            )
        if not issubclass(cls, BaseProvider):
            raise ValueError(f"{cls.__name__} must extend BaseProvider.")

        _PROVIDERS_REGISTRY[name] = cls
        return cls

    return register_provider_cls


def get_provider(name, **kwargs):
    if name not in _PROVIDERS_REGISTRY:
        raise ValueError(f"Unknown provider '{name}'.")
    return _PROVIDERS_REGISTRY[name](**kwargs)
