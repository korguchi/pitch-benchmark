from importlib import import_module
from typing import Dict, Type, Optional, List
from .base import PitchAlgorithm

# Algorithm metadata - maps names to (module_name, class_name, required_packages)
_ALGORITHM_METADATA = {
    # "CREPE": ("crepe", "CREPEPitchAlgorithm", ["crepe", "tensorflow"]),
    "PENN": ("penn", "PENNPitchAlgorithm", ["penn"]),
    "Praat": ("praat", "PraatPitchAlgorithm", ["praat-parselmouth"]),
    "RAPT": ("rapt", "RAPTPitchAlgorithm", ["pysptk"]),
    "SWIPE": ("swipe", "SWIPEPitchAlgorithm", ["pysptk"]),
    "TorchCREPE": ("torchcrepe", "TorchCREPEPitchAlgorithm", ["torchcrepe", "torch"]),
    "YAAPT": ("yaapt", "YAAPTPitchAlgorithm", ["AMFM-decompy"]),
    "pYIN": ("pyin", "pYINPitchAlgorithm", ["librosa"]),
    "BasicPitch": ("basicpitch", "BasicPitchPitchAlgorithm", ["basic-pitch"]),
    "SwiftF0": ("swiftf0", "SwiftF0PitchAlgorithm", ["swift-f0"]),
}

__all__ = ["PitchAlgorithm", "get_algorithm", "register_algorithm", "list_algorithms"]
_REGISTRY: Dict[str, Type[PitchAlgorithm]] = {}
_IMPORT_ERRORS: Dict[str, str] = {}


def _try_import_algorithm(
    name: str, module_name: str, class_name: str
) -> Optional[Type[PitchAlgorithm]]:
    try:
        module = import_module(f".{module_name}", package=__package__)
        cls = getattr(module, class_name)
        globals()[class_name] = cls  # Make available at package level
        __all__.append(class_name)
        return cls
    except ImportError as e:
        _IMPORT_ERRORS[name] = str(e)
    except AttributeError as e:
        _IMPORT_ERRORS[name] = f"Class {class_name} not found: {str(e)}"
    return None


# Initialize all algorithms
for name, (module_name, class_name, _) in _ALGORITHM_METADATA.items():
    if cls := _try_import_algorithm(name, module_name, class_name):
        _REGISTRY[name] = cls


def get_algorithm_dependencies(name: str) -> list:
    """Get required packages for a specific algorithm"""
    if name not in _ALGORITHM_METADATA:
        raise ValueError(f"Unknown algorithm: {name}")
    return _ALGORITHM_METADATA[name][2]


def get_algorithm(
    name: str, fail_silently: bool = False
) -> Optional[Type[PitchAlgorithm]]:
    """Get algorithm class by name

    Args:
        name: Name of the algorithm to get
        fail_silently: If True, returns None when algorithm is unavailable instead of raising an error

    Returns:
        The algorithm class if available, None if not available and fail_silently is True

    Raises:
        ImportError: If algorithm is not available and fail_silently is False
        ValueError: If algorithm name is unknown
    """
    if name not in _ALGORITHM_METADATA:
        if fail_silently:
            return None
        raise ValueError(f"Unknown algorithm: {name}")

    if name not in _REGISTRY:
        if fail_silently:
            return None
        available = list(_REGISTRY.keys())
        if name in _IMPORT_ERRORS:
            deps = get_algorithm_dependencies(name)
            raise ImportError(
                f"Algorithm '{name}' requires packages: {deps}\n"
                f"Error: {_IMPORT_ERRORS[name]}\n"
                f"Available algorithms: {available}"
            )
        raise ValueError(f"Unknown algorithm: {name}. Available: {available}")
    return _REGISTRY[name]


def get_available_algorithms() -> List[str]:
    """Get list of available algorithm names"""
    return list(_REGISTRY.keys())


def register_algorithm(name: str, algorithm_class: Type[PitchAlgorithm]):
    """Register a custom algorithm"""
    if not issubclass(algorithm_class, PitchAlgorithm):
        raise TypeError("Algorithm must subclass PitchAlgorithm")
    _REGISTRY[name] = algorithm_class
    if name not in __all__:
        __all__.append(name)


def list_algorithms(verbose: bool = False) -> Dict[str, str]:
    """List available algorithms with status"""
    results = {}
    for name in _ALGORITHM_METADATA:
        if name in _REGISTRY:
            results[name] = "available"
        else:
            deps = get_algorithm_dependencies(name)
            results[name] = (
                f"unavailable (requires: {deps}, "
                f"error: {_IMPORT_ERRORS.get(name, 'unknown')})"
            )
    return results
