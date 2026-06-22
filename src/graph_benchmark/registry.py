from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}

    def register(self, key: str) -> Callable[[T], T]:
        def decorator(item: T) -> T:
            if key in self._items:
                raise KeyError(f"'{key}' is already registered in '{self.name}'.")

            self._items[key] = item
            return item

        return decorator

    def get(self, key: str) -> T:
        if key not in self._items:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"Unknown key '{key}' in registry '{self.name}'. Available: {available}")

        return self._items[key]

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


DATASET_REGISTRY: Registry[Callable] = Registry("dataset")
ALGORITHM_REGISTRY: Registry[Callable] = Registry("algorithm")
EXPERIMENT_REGISTRY: Registry[Callable] = Registry("experiment")
SEED_SELECTOR_REGISTRY: Registry[Callable] = Registry("seed_selector")
