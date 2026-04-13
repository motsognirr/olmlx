# Partial stub for mlx.nn.layers.base.Module.
#
# Why this stub exists: mlx ships a runtime `Module.__getattr__` whose else
# branch falls through and implicitly returns None. Pyright therefore infers
# every `model.<attr>` access as Optional, which floods reportArgumentType
# / reportOptionalMemberAccess across the project.
#
# This stub overrides only the attribute-access protocol and a few
# commonly-used methods, leaving everything else for pyright to recover
# from runtime introspection of subclasses.

from typing import Any, Callable

import mlx.core as mx

class Module(dict):
    training: bool
    state: Any

    __call__: Callable[..., Any]

    def __init__(self) -> None: ...
    def __getattr__(self, key: str) -> Any: ...
    def __setattr__(self, key: str, val: Any) -> None: ...
    def parameters(self) -> dict[str, Any]: ...
    def trainable_parameters(self) -> dict[str, Any]: ...
    def children(self) -> dict[str, Any]: ...
    def leaf_modules(self) -> dict[str, Any]: ...
    def named_modules(self) -> list[tuple[str, "Module"]]: ...
    def modules(self) -> list["Module"]: ...
    def update(self, parameters: dict[str, Any]) -> "Module": ...  # type: ignore[override]
    def update_modules(self, modules: dict[str, Any]) -> "Module": ...
    def apply(
        self,
        map_fn: Callable[[mx.array], mx.array],
        filter_fn: Callable[..., bool] | None = ...,
    ) -> "Module": ...
    def apply_to_modules(
        self, apply_fn: Callable[[str, "Module"], Any]
    ) -> "Module": ...
    def filter_and_map(
        self,
        filter_fn: Callable[..., bool],
        map_fn: Callable[[Any], Any] | None = ...,
        is_leaf_fn: Callable[..., bool] | None = ...,
    ) -> dict[str, Any]: ...
    def freeze(
        self,
        *,
        recurse: bool = ...,
        keys: str | list[str] | None = ...,
        strict: bool = ...,
    ) -> "Module": ...
    def unfreeze(
        self,
        *,
        recurse: bool = ...,
        keys: str | list[str] | None = ...,
        strict: bool = ...,
    ) -> "Module": ...
    def train(self, mode: bool = ...) -> "Module": ...
    def eval(self) -> "Module": ...
    def load_weights(
        self,
        file_or_weights: str | list[tuple[str, mx.array]],
        strict: bool = ...,
    ) -> "Module": ...
    def save_weights(self, file: str) -> None: ...
    def is_module(self, value: Any) -> bool: ...
    def set_dtype(
        self,
        dtype: Any,
        predicate: Callable[[Any], bool] | None = ...,
    ) -> None: ...
    @staticmethod
    def trainable_parameter_filter(*args: Any, **kwargs: Any) -> bool: ...
    @staticmethod
    def valid_child_filter(*args: Any, **kwargs: Any) -> bool: ...
    @staticmethod
    def valid_parameter_filter(*args: Any, **kwargs: Any) -> bool: ...
