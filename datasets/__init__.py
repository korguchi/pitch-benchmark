from .base import PitchDataset
from .ptdb import PitchDatasetPTDB
from .mdb import PitchDatasetMDBStemSynth
from .nsynth import PitchDatasetNSynth

__all__ = [
    "PitchDataset",
    "PitchDatasetPTDB",
    "PitchDatasetMDBStemSynth",
    "PitchDatasetNSynth",
]

_REGISTRY = {
    "PTDB": PitchDatasetPTDB,
    "NSynth": PitchDatasetNSynth,
    "MDBStemSynth": PitchDatasetMDBStemSynth,
}


def get_dataset(name: str):
    """Get dataset class by name"""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def register_dataset(name: str, dataset_class):
    """Register a new dataset"""
    _REGISTRY[name] = dataset_class


def list_datasets():
    """List available datasets"""
    return list(_REGISTRY.keys())
