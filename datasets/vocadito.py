import torch
from pathlib import Path
import torchaudio
import numpy as np
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
        self.annot_f0_dir = self.root_dir / "Annotations" / "F0"
        self.annot_notes_dir = self.root_dir / "Annotations" / "Notes"
        self.metadata_path = self.root_dir / "vocadito_metadata.csv"

        if not all(
            [
                self.audio_dir.exists(),
                self.annot_f0_dir.exists(),
                self.annot_notes_dir.exists(),
                self.metadata_path.exists(),
            ]
        ):
            raise FileNotFoundError(
                "A required directory or file (Audio, Annotations/F0, Annotations/Notes, or vocadito_metadata.csv) was not found."
            )

        # Load metadata for grouping
        try:
            self.metadata = pd.read_csv(self.metadata_path)
            self.metadata.set_index("track_id", inplace=True)
        except Exception as e:
            raise IOError(f"Error loading metadata file {self.metadata_path}: {str(e)}")

        self.use_cache = use_cache
        self.data_cache: Dict[
            int, Dict[str, Union[torch.Tensor, Path, List[Dict]]]
        ] = {}

        # Find all valid wav-annotation pairs
        self.wav_f0_notes_pairs = self._find_wav_f0_notes_pairs()
        if not self.wav_f0_notes_pairs:
            raise ValueError(
                f"No valid wav-F0-notes annotation pairs found in '{root_dir}'"
            )

    def _find_wav_f0_notes_pairs(self) -> List[Tuple[Path, Path, Path]]:
        """
        Find matching WAV, F0 CSV, and Notes CSV annotation file pairs.

        This method identifies audio files and their corresponding F0 and note annotation
        files. It specifically looks for 'vocadito_*_notesA1.csv' as the primary
        note annotation source. Pairs are only included if all three files exist.

        Returns:
            List[Tuple[Path, Path, Path]]: A list of tuples, where each tuple contains
                                           (audio_file_path, f0_csv_path, notes_csv_path).
        """
        pairs = []
        for wav_path in sorted(self.audio_dir.glob("vocadito_*.wav")):
            file_stem = wav_path.stem  # e.g., 'vocadito_1'
            f0_csv_path = self.annot_f0_dir / f"{file_stem}_f0.csv"
            # Using Annotator 1's notes for consistency in benchmarking
            notes_csv_path = (
                self.annot_notes_dir / f"{file_stem}_notesA1.csv"
            )  # NEW: Notes CSV path

            if f0_csv_path.exists() and notes_csv_path.exists():
                pairs.append((wav_path, f0_csv_path, notes_csv_path))  # NEW:
            else:
                if not f0_csv_path.exists():
                    print(
                        f"Warning: Skipping {file_stem} due to missing F0 annotation: {f0_csv_path}"
                    )
                if not notes_csv_path.exists():
                    print(
                        f"Warning: Skipping {file_stem} due to missing Notes annotation: {notes_csv_path}"
                    )
        return pairs

    def _load_f0_annotation(self, csv_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load F0 and compute periodicity from a vocadito F0 CSV annotation.

        The CSV files contain timestamps and F0 values in Hz.
        A value of 0.0 Hz indicates that no F0 is present (unvoiced).

        Args:
            csv_path (Path): Path to the F0 CSV annotation file.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - pitch (torch.Tensor): F0 values in Hz.
                - periodicity (torch.Tensor): Binary periodicity (1.0 for voiced, 0.0 for unvoiced).
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

    def _load_notes_annotation(self, csv_path: Path) -> List[Dict]:
        """Load and convert annotations to standardized format."""
        notes = []
        try:
            data = pd.read_csv(csv_path, header=None).values
            for row in data:
                start = float(row[0])
                pitch_hz = float(row[1])
                duration = float(row[2])
                end = start + duration

                # Convert Hz to MIDI
                midi_pitch = 12 * (np.log2(pitch_hz / 440.0)) + 69

                notes.append(
                    {
                        "start": start,
                        "end": end,
                        "midi_pitch": float(midi_pitch),
                    }
                )
        except Exception as e:
            print(f"Warning: Error loading {csv_path}: {str(e)}")
            return []
        return notes

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.wav_f0_notes_pairs)

    def get_group(self, idx: int) -> str:
        """Return the group identifier for a sample, which is the singer_id."""
        # Extract track_id from filename, e.g., 'vocadito_1' -> 1
        wav_path = self.wav_f0_notes_pairs[idx][0]
        track_id = int(wav_path.stem.split("_")[1])
        # Look up singer_id from the metadata
        singer_id = self.metadata.loc[track_id, "singer_id"]
        return str(singer_id)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Path]]:
        """
        Get a sample from the dataset, including audio, F0, periodicity, and musical notes.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict[str, Union[torch.Tensor, Path, List[Dict]]]: A dictionary containing:
                - "audio" (torch.Tensor): The processed audio waveform.
                - "pitch" (torch.Tensor): The processed ground truth F0 values in Hz.
                - "periodicity" (torch.Tensor): The processed ground truth periodicity values.
                - "notes" (List[Dict]): A list of dictionaries, each representing a musical note
                                         with 'start', 'pitch_hz', and 'duration' keys.
                - "wav_path" (Path): The original path to the audio file.
        """
        if not 0 <= idx < len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        if idx in self.data_cache and self.use_cache:
            return self.data_cache[idx]
        else:
            wav_path, f0_csv_path, notes_csv_path = self.wav_f0_notes_pairs[idx]

            try:
                waveform, sr = torchaudio.load(wav_path)
                waveform = waveform.squeeze()
            except Exception as e:
                raise IOError(f"Error loading audio file {wav_path}: {str(e)}")

            pitch, periodicity = self._load_f0_annotation(f0_csv_path)
            notes = self._load_notes_annotation(notes_csv_path)

            # Process the sample using the method from the base class
            waveform, pitch, periodicity, notes = self.process_sample(
                waveform, pitch, periodicity, sr, notes
            )

            sample_dict = {
                "audio": waveform,
                "pitch": pitch,
                "periodicity": periodicity,
                "notes": notes,
                "wav_path": wav_path,
            }

            if self.use_cache:
                self.data_cache[idx] = sample_dict

            return sample_dict
