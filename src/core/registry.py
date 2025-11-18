# src/core/registry.py
from __future__ import annotations
from typing import Any, Callable, Dict

class Registry:
    def __init__(self, name: str):
        self._name = name
        self._obj_map: Dict[str, Any] = {}

    def register(self, name: str) -> Callable:
        """ As decorator to register an object with a given name
        """
        def decorator(obj: Any) -> Any:
            if name in self._obj_map:
                raise KeyError(f"{name} already registered in {self._name}")
            self._obj_map[name] = obj
            return obj
        return decorator

    def get(self, name: str) -> Any:
        if name not in self._obj_map:
            raise KeyError(
                f"{name} is not registered in {self._name}. "
                f"Available keys: {list(self._obj_map.keys())}"
            )
        return self._obj_map[name]

    def create(self, name: str, *args, **kwargs) -> Any:
        cls = self.get(name)
        return cls(*args, **kwargs)

MODEL_REGISTRY = Registry("model")
DATAMODULE_REGISTRY = Registry("datamodule")
LOSS_REGISTRY = Registry("loss")
