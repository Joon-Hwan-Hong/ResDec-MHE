"""
Conftest for training tests.

Applies torchvision mock to work around broken torchvision C extension
in this environment. Lightning imports torchmetrics which imports torchvision,
but torchvision._C.so has undefined symbols. Since training tests don't need
torchvision, we mock it before lightning is imported.
"""

import sys
import types


class _MockModule(types.ModuleType):
    """Mock module that returns sub-mocks for any attribute access."""

    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []
        self.__file__ = "<mock>"
        self.__version__ = "0.25.0"
        self.__all__ = []

    def __getattr__(self, name):
        sub = _MockModule(f"{self.__name__}.{name}")
        setattr(self, name, sub)
        return sub

    def __call__(self, *args, **kwargs):
        return None


# Only apply mock if torchvision is not already successfully imported
if "torchvision" not in sys.modules:
    _tv_names = [
        "torchvision", "torchvision._meta_registrations",
        "torchvision.datasets", "torchvision.io", "torchvision.models",
        "torchvision.ops", "torchvision.transforms", "torchvision.utils",
        "torchvision.transforms.functional", "torchvision.extension",
    ]
    for mod_name in _tv_names:
        sys.modules[mod_name] = _MockModule(mod_name)
    sys.modules["torchvision"].extension._has_ops = lambda: False
