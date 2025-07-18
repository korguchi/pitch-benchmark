import torch
from pathlib import Path
import torchaudio
from typing import Dict, List, Tuple, Union
import pandas as pd
from .base import PitchDataset


class PitchDatasetVocadito(PitchDataset):
    """
    Implementation of PitchDataset for the vocadito dataset.

    The vocadito dataset contains 40 short excerpts of solo, monophonic singing.
    For more details, see the technical report: https://doi.org/10.5281/zenodo.4619711

    Args:
        root_dir (str): Root directory of the vocadito dataset.
        use_cache (bool, optional): Whether to cache loaded data. Defaults to True.
        **kwargs: Additional arguments passed to the base PitchDataset.
    """

    def __init__(self, root_dir: str, use_cache: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory '{root_dir}' does not exist")

        # Set up directory paths according to vocadito structure
        self.audio_dir = self.root_dir / "Audio"
        self.annot_dir = self.root_dir / "Annotations" / "F0"
        self.metadata_path = self.root_dir / "vocadito_metadata.csv"

        if not all(
            [
                self.audio_dir.exists(),
                self.annot_dir.exists(),
                self.metadata_path.exists(),
            ]
        ):
            raise FileNotFoundError(
                "A required directory or file (Audio, Annotations/F0, or vocadito_metadata.csv) was not found."
            )

        # Load metadata for grouping
        try:
            self.metadata = pd.read_csv(self.metadata_path)
            # Set track_id as index for easy lookup
            self.metadata.set_index("track_id", inplace=True)
        except Exception as e:
            raise IOError(f"Error loading metadata file {self.metadata_path}: {str(e)}")

        self.use_cache = use_cache
        self.data_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # Find all valid wav-annotation pairs
        self.wav_f0_pairs = self._find_wav_f0_pairs()
        if not self.wav_f0_pairs:
            raise ValueError(f"No valid wav-annotation pairs found in '{root_dir}'")

    def _find_wav_f0_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching WAV and F0 CSV annotation file pairs."""
        pairs = []
        # Audio files are named like 'vocadito_1.wav', 'vocadito_2.wav', etc.
        for wav_path in sorted(self.audio_dir.glob("vocadito_*.wav")):
            # F0 files are named like 'vocadito_1_f0.csv'
            csv_path = self.annot_dir / f"{wav_path.stem}_f0.csv"
            if csv_path.exists():
                pairs.append((wav_path, csv_path))
        return pairs

    def _load_f0_annotation(self, csv_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load F0 and compute periodicity from a vocadito CSV annotation.

        The CSV files contain timestamps and F0 values in Hz.
        A value of 0.0 Hz indicates that no F0 is present (unvoiced).
        """
        try:
            data = pd.read_csv(csv_path, header=None).values
            # Column 1 (index 1) contains the F0 values
            pitch = torch.from_numpy(data[:, 1]).float()
            # Periodicity is true where pitch is greater than 0
            periodicity = (pitch > 0).float()
            return pitch, periodicity
        except Exception as e:
            raise IOError(f"Error loading annotation file {csv_path}: {str(e)}")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.wav_f0_pairs)

    def get_group(self, idx: int) -> str:
        """Return the group identifier for a sample, which is the singer_id."""
        wav_path = self.wav_f0_pairs[idx][0]
        # Extract track_id from filename, e.g., 'vocadito_1' -> 1
        track_id = int(wav_path.stem.split("_")[1])
        # Look up singer_id from the metadata
        singer_id = self.metadata.loc[track_id, "singer_id"]
        return str(singer_id)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Path]]:
        """Get a sample from the dataset."""
        if not 0 <= idx < len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        if idx in self.data_cache and self.use_cache:
            waveform, pitch, periodicity = self.data_cache[idx]
        else:
            wav_path, csv_path = self.wav_f0_pairs[idx]

            try:
                waveform, sr = torchaudio.load(wav_path)
                waveform = waveform.squeeze()
            except Exception as e:
                raise IOError(f"Error loading audio file {wav_path}: {str(e)}")

            pitch, periodicity = self._load_f0_annotation(csv_path)

            # Process the sample using the method from the base class
            waveform, pitch, periodicity = self.process_sample(
                waveform, pitch, periodicity, sr
            )

            if self.use_cache:
                self.data_cache[idx] = (waveform, pitch, periodicity)

        return {
            "audio": waveform,
            "pitch": pitch,
            "periodicity": periodicity,
            "wav_path": self.wav_f0_pairs[idx][0],
        }
