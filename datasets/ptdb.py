import torch
from pathlib import Path
import torchaudio
from typing import Dict, List, Tuple, Union
import numpy as np
from .base import PitchDataset


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

        # Find all valid wav-f0 pairs
        self.wav_f0_pairs = self._find_wav_f0_pairs()
        if not self.wav_f0_pairs:
            raise ValueError(f"No valid wav-f0 pairs found in '{root_dir}'")

    def _find_wav_f0_pairs(self) -> List[Tuple[Path, Path]]:
        """Finds matching WAV and F0 file pairs in the dataset."""
        pairs = []
        for gender in ["MALE", "FEMALE"]:
            mic_dir = self.root_dir / gender / "MIC"
            ref_dir = self.root_dir / gender / "REF"

            if not mic_dir.exists() or not ref_dir.exists():
                continue

            for wav_path in mic_dir.rglob("*.wav"):
                f0_path = ref_dir / wav_path.relative_to(mic_dir).with_name(
                    wav_path.name.replace("mic_", "ref_").replace(".wav", ".f0")
                )
                if f0_path.exists():
                    pairs.append((wav_path, f0_path))
        return pairs

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
