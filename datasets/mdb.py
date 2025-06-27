import torch
from pathlib import Path
import torchaudio
from typing import Dict, List, Tuple, Union
import pandas as pd
from .base import PitchDataset


class PitchDatasetMDBStemSynth(PitchDataset):
    """
    Implementation of PitchDataset for the MDB-stem-synth dataset.

    The dataset contains resynthesized solo stems from MedleyDB with perfect F0 annotations.
    Annotations are provided at a hop size of 128/44100 seconds (~2.9 ms).

    Args:
        root_dir (str): Root directory of the MDB-stem-synth dataset
        use_cache (bool, optional): Whether to cache loaded data. Defaults to True
        **kwargs: Additional arguments passed to PitchDataset
    """

    def __init__(self, root_dir: str, use_cache: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory '{root_dir}' does not exist")

        # Set up directory paths according to MDB-stem-synth structure
        self.audio_dir = self.root_dir / "audio_stems"
        self.annot_dir = self.root_dir / "annotation_stems"

        if not self.audio_dir.exists() or not self.annot_dir.exists():
            raise FileNotFoundError(
                "Audio stems or annotation stems directory not found"
            )

        self.use_cache = use_cache
        self.data_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # Find all valid wav-annotation pairs
        self.wav_f0_pairs = self._find_wav_f0_pairs()
        if not self.wav_f0_pairs:
            raise ValueError(f"No valid wav-annotation pairs found in '{root_dir}'")

    def _find_wav_f0_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching WAV and CSV annotation file pairs in the dataset."""
        pairs = []
        for wav_path in self.audio_dir.glob("*.RESYN.wav"):
            csv_path = self.annot_dir / wav_path.name.replace(".wav", ".csv")
            if csv_path.exists():
                pairs.append((wav_path, csv_path))
        return sorted(pairs)

    def _load_f0_annotation(self, csv_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load F0 and compute periodicity from CSV annotation.

        The CSV files contain timestamps and F0 values in Hz.
        Silence is indicated as 0 Hz.
        """
        try:
            data = pd.read_csv(csv_path, header=None).values
            pitch = torch.from_numpy(data[:, 1]).float()  # F0 values are in column 1
            periodicity = (pitch > 0).float()  # Non-zero F0 indicates voiced frames
            return pitch, periodicity
        except Exception as e:
            raise IOError(f"Error loading annotation file {csv_path}: {str(e)}")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.wav_f0_pairs)

    def get_group(self, idx: int) -> str:
        """Return group identifier for sample (artist name)"""
        file_path = self.wav_f0_pairs[idx][0]
        # Extract artist from filename (first part before underscore)
        return file_path.stem.split("_")[0]

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Path]]:
        """Get a sample from the dataset."""
        if not 0 <= idx < len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        if idx not in self.data_cache or not self.use_cache:
            wav_path, csv_path = self.wav_f0_pairs[idx]

            try:
                waveform, sr = torchaudio.load(wav_path)
                waveform = waveform.squeeze()
            except Exception as e:
                raise IOError(f"Error loading audio file {wav_path}: {str(e)}")

            pitch, periodicity = self._load_f0_annotation(csv_path)

            # Process the sample
            waveform, pitch, periodicity = self.process_sample(
                waveform, pitch, periodicity, sr
            )

            if self.use_cache:
                self.data_cache[idx] = (waveform, pitch, periodicity)
        else:
            waveform, pitch, periodicity = self.data_cache[idx]

        return {
            "audio": waveform,
            "pitch": pitch,
            "periodicity": periodicity,
            "wav_path": self.wav_f0_pairs[idx][0],
        }
