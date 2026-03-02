from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any

from datasets import Dataset


class DatasetAdapter(ABC):
    @abstractmethod
    def adapt_example(self, example: dict[str, Any], index: int) -> dict[str, Any]:
        pass

    def adapt_dataset(self, dataset: Dataset) -> Dataset:
        def _map_fn(example, idx):
            return self.adapt_example(example, idx)

        return dataset.map(_map_fn, with_indices=True)


def load_dataset_adapter(adapter_name: str):
    if adapter_name in ("default", "DefaultCodeDatasetAdapter"):
        from .default_adapter import DefaultCodeDatasetAdapter

        return DefaultCodeDatasetAdapter()

    if ":" in adapter_name:
        module_name, class_name = adapter_name.split(":", 1)
    elif "." in adapter_name:
        module_name, class_name = adapter_name.rsplit(".", 1)
    else:
        raise ValueError(
            "dataset_adapter must be 'default' or a class path like 'my_pkg.module:MyAdapter' or "
            "'my_pkg.module.MyAdapter'."
        )

    module = import_module(module_name)
    adapter_cls = getattr(module, class_name)
    adapter = adapter_cls()
    if not isinstance(adapter, DatasetAdapter):
        raise TypeError(f"{adapter_name} is not a DatasetAdapter.")
    return adapter

