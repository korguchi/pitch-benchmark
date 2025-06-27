import torch
from pathlib import Path
import torchaudio
from typing import Dict, List, Tuple, Union, Set
import numpy as np
from .base import PitchDataset

CORRUPT_FILES_PATH = Path(__file__).parent / "PTDB_CORRUPT_FILES.txt"


class PitchDatasetPTDB(PitchDataset):
    """
    Implementation of PitchDataset for the Pitch Tracking Database (PTDB).

    Args:
        root_dir (str): Root directory of the PTDB dataset
        use_cache (bool, optional): Whether to cache loaded data. Defaults to True
        **kwargs: Additional arguments passed to PitchDataset
    """

    def __init__(self, root_dir: str, use_cache: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory '{root_dir}' does not exist")

        self.use_cache = use_cache
        self.data_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.corrupt_files = self._load_corrupt_files()

        # Find all valid wav-f0 pairs
        self.wav_f0_pairs = self._find_wav_f0_pairs()
        if not self.wav_f0_pairs:
            raise ValueError(f"No valid wav-f0 pairs found in '{root_dir}'")

    def _load_corrupt_files(self) -> Set[str]:
        """
        Load the list of corrupt files identified by quality metrics.

        The corrupt files listed in CORRUPT_FILES_PATH were identified through
        analysis of pitch tracking and signal quality metrics. These files
        exhibited severe issues such as low harmonic mean across accuracy metrics,
        high pitch deviation (cents error), poor recall, or suspicious error patterns.

        347 files (7.4% of the dataset) were identified as corrupt, ordered by descending severity.

        Returns:
            Set[str]: A set of basenames (filenames only, no paths) for the corrupt files.

        Raises:
            FileNotFoundError: If CORRUPT_FILES_PATH does not exist.
        """
        try:
            with open(CORRUPT_FILES_PATH, "r") as f:
                return {line.strip() for line in f if line.strip()}
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Corrupt files list not found at {CORRUPT_FILES_PATH}. "
                "Please ensure PTDB_CORRUPT_FILES.txt exists in the datasets directory."
            )

    def _find_wav_f0_pairs(self) -> List[Tuple[Path, Path]]:
        """Finds matching WAV and F0 file pairs in the dataset."""
        pairs = []
        for gender in ["MALE", "FEMALE"]:
            mic_dir = self.root_dir / gender / "MIC"
            ref_dir = self.root_dir / gender / "REF"

            if not mic_dir.exists() or not ref_dir.exists():
                continue

            for wav_path in mic_dir.rglob("*.wav"):
                if wav_path.name in self.corrupt_files:
                    continue
                f0_path = ref_dir / wav_path.relative_to(mic_dir).with_name(
                    wav_path.name.replace("mic_", "ref_").replace(".wav", ".f0")
                )
                if f0_path.exists():
                    pairs.append((wav_path, f0_path))
        return pairs

    def get_group(self, idx: int) -> str:
        wav_path = self.wav_f0_pairs[idx][0]
        parts = wav_path.stem.split("_")
        return parts[1] if len(parts) >= 2 else "unknown"  # e.g., "F08"

    def __len__(self) -> int:
        return len(self.wav_f0_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Path]]:
        if not 0 <= idx < len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        if idx not in self.data_cache or not self.use_cache:
            wav_path, f0_path = self.wav_f0_pairs[idx]

            try:
                waveform, sr = torchaudio.load(wav_path)
                waveform = waveform.squeeze()
            except Exception as e:
                raise IOError(f"Error loading audio file {wav_path}: {str(e)}")

            try:
                pitch = torch.from_numpy(np.loadtxt(f0_path)[:, 0]).float()
                periodicity = (pitch > 0).float()
            except Exception as e:
                raise IOError(f"Error loading F0 file {f0_path}: {str(e)}")

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
