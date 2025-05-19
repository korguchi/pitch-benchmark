import torch
import json
from pathlib import Path
import torchaudio
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class PitchDataset(ABC, torch.utils.data.Dataset):
    """
    Abstract base class for audio datasets with pitch and periodicity processing.

    Provides core functionality for audio processing, pitch validation, and resampling
    while defining an interface that derived classes must implement.

    Args:
        sample_rate (int): Target sample rate in Hz
        hop_size (int): Number of audio samples between consecutive frames
        fmin (float, optional): Minimum frequency in Hz. Values below will be clipped to this. Defaults to 20.0
        fmax (float, optional): Maximum frequency in Hz. Values above will be clipped to this. Defaults to 2000.0
        normalize_audio (bool, optional): Whether to normalize audio to [-1, 1]. Defaults to True
    """

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        fmin: float = 20.0,
        fmax: float = 2000.0,
        normalize_audio: bool = True,
    ):
        super().__init__()
        self._validate_init_params(sample_rate, hop_size, fmin, fmax)

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.normalize_audio = normalize_audio

    def _validate_init_params(
        self, sample_rate: int, hop_size: int, fmin: float, fmax: float
    ) -> None:
        """Validates initialization parameters."""
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        if hop_size <= 0:
            raise ValueError(f"Hop size must be positive, got {hop_size}")
        if fmin >= fmax:
            raise ValueError(f"fmin ({fmin} Hz) must be less than fmax ({fmax} Hz)")
        if fmin < 0:
            raise ValueError(f"fmin ({fmin} Hz) must be non-negative")
        if fmax > sample_rate / 2:
            raise ValueError(
                f"fmax ({fmax} Hz) must not exceed Nyquist frequency ({sample_rate / 2} Hz)"
            )

    def _validate_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Validates and normalizes audio data.

        Args:
            audio (torch.Tensor): Input audio tensor

        Returns:
            torch.Tensor: Validated and normalized audio
        """
        if audio.dim() not in {1, 2}:
            raise ValueError(f"Audio must be 1D or 2D, got {audio.dim()}D")

        # Clean up audio values
        audio = torch.nan_to_num(audio, nan=0)

        if torch.all(audio == 0):
            raise ValueError(f"Silent audio!")

        if self.normalize_audio:
            max_abs = audio.abs().max()
            if max_abs > 1:  # Normalize only if the range exceeds -1 to 1
                audio = audio / max_abs

        return audio.clamp(-1.0, 1.0)

    def _validate_pitch(
        self, pitch: torch.Tensor, periodicity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Validates and processes pitch and periodicity values.

        Args:
            pitch (torch.Tensor): Pitch values
            periodicity (torch.Tensor): Periodicity values

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Processed pitch and periodicity values.
                Pitch values are clipped to [fmin, fmax] range.
        """
        if pitch.shape != periodicity.shape:
            raise ValueError(
                f"Pitch and periodicity shapes must match: {pitch.shape} vs {periodicity.shape}"
            )

        # Clean up pitch values
        pitch = torch.nan_to_num(pitch, nan=self.fmin)
        # Clip pitch values to valid range
        pitch = torch.clamp(pitch, self.fmin, self.fmax)

        # Ensure periodicity is binary
        periodicity = torch.round(periodicity).clamp(0, 1)

        if torch.all(periodicity == 0) or torch.all(pitch == 0):
            raise ValueError(f"No pitch!")

        # Zero out pitch where periodicity is 0
        pitch = pitch * periodicity

        return pitch, periodicity

    def process_sample(
        self,
        audio: torch.Tensor,
        pitch: torch.Tensor,
        periodicity: torch.Tensor,
        orig_sr: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes a single audio sample with its corresponding pitch and periodicity.

        Args:
            audio (torch.Tensor): Audio waveform
            pitch (torch.Tensor): Pitch values
            periodicity (torch.Tensor): Periodicity values
            orig_sr (int): Original sample rate of the audio

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Processed audio, pitch, and periodicity
        """
        # Ensure consistent dimensions
        audio = audio.squeeze()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Resample audio if needed
        if orig_sr != self.sample_rate:
            audio = torchaudio.functional.resample(
                waveform=audio, orig_freq=orig_sr, new_freq=self.sample_rate
            )

        # Basic validation
        audio = self._validate_audio(audio)

        # Calculate target length for pitch and periodicity
        target_length = 1 + audio.size(-1) // self.hop_size

        if target_length > 0:
            # Interpolate pitch and periodicity to match target length
            pitch = F.interpolate(
                pitch.view(1, 1, -1),
                size=target_length,
                mode="linear",
                align_corners=True,
            ).squeeze()

            periodicity = F.interpolate(
                periodicity.view(1, 1, -1), size=target_length, mode="nearest"
            ).squeeze()

        # Validate pitch and periodicity
        pitch, periodicity = self._validate_pitch(pitch, periodicity)

        return audio.squeeze(0), pitch, periodicity

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Path]]:
        """Retrieves a single item from the dataset."""
        pass


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


class PitchDatasetNSynth(PitchDataset):
    """
    Dataset implementation for the NSynth (Neural Audio Synthesis) dataset.

    This class handles loading and processing of the NSynth dataset, which contains
    musical notes from various instruments. It provides filtering capabilities based
    on instrument types, families, note qualities, and pitch frequency ranges.

    Args:
        root_dir (str): Path to NSynth dataset directory containing examples.json and audio files
        instrument_sources (Optional[List[str]]): Filter by instrument sources
            Valid options: ["acoustic", "electronic", "synthetic"]
        instrument_families (Optional[List[str]]): Filter by instrument families
            Valid options: ["bass", "brass", "flute", "guitar", "keyboard", "mallet",
                          "organ", "reed", "string", "synth_lead", "vocal"]
        qualities (Optional[List[str]]): Filter by note qualities
            Valid options: ["bright", "dark", "distortion", "fast_decay", "long_release",
                          "multiphonic", "nonlinear_env", "percussive", "reverb", "tempo-synced"]
        use_cache (bool): Whether to cache loaded audio in memory. Defaults to True
        sample_rate (int): Target sample rate for the audio. Passed to parent class
        hop_size (int): Number of audio samples between consecutive frames. Passed to parent class
        fmin (float, optional): Minimum frequency in Hz. Notes below this frequency will be filtered out.
            Defaults to 27.5 Hz (A0)
        fmax (float, optional): Maximum frequency in Hz. Notes above this frequency will be filtered out.
            Defaults to 4186.0 Hz (C8)
        silence_threshold_db (float, optional): Threshold in dB below which audio is considered silent.
            Defaults to -40.0
        **kwargs: Additional arguments passed to PitchDataset base class

    Raises:
        ValueError: If no examples match the specified criteria or if invalid parameters are provided
        IOError: If there are errors loading the dataset metadata
    """

    VALID_SOURCES = {"acoustic", "electronic", "synthetic"}
    VALID_FAMILIES = {
        "bass",
        "brass",
        "flute",
        "guitar",
        "keyboard",
        "mallet",
        "organ",
        "reed",
        "string",
        "synth_lead",
        "vocal",
    }
    VALID_QUALITIES = {
        "bright",
        "dark",
        "distortion",
        "fast_decay",
        "long_release",
        "multiphonic",
        "nonlinear_env",
        "percussive",
        "reverb",
        "tempo-synced",
    }

    def __init__(
        self,
        root_dir: str,
        instrument_sources: Optional[List[str]] = None,
        instrument_families: Optional[List[str]] = None,
        qualities: Optional[List[str]] = None,
        use_cache: bool = True,
        sample_rate: int = 16000,
        hop_size: int = 160,
        fmin: float = 27.5,  # A0 (MIDI note 21)
        fmax: float = 4186.0,  # C8 (MIDI note 108)
        silence_threshold_db: float = -40.0,
        **kwargs,
    ):
        # Initialize parent class with NSynth-specific defaults
        super().__init__(
            sample_rate=sample_rate, hop_size=hop_size, fmin=fmin, fmax=fmax, **kwargs
        )

        self.root_dir = Path(root_dir)
        self.use_cache = use_cache
        self.data_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # Parameters for audio energy detection
        self.silence_threshold_db = silence_threshold_db

        # Load metadata
        json_path = self.root_dir / "examples.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {json_path}")

        with open(json_path, "r") as f:
            self.metadata = json.load(f)

        # Filter and prepare examples
        self.examples = self._prepare_examples(
            instrument_sources, instrument_families, qualities, fmin, fmax
        )

        if not self.examples:
            raise ValueError(
                "No examples found matching the specified criteria. "
                "Try adjusting the filter parameters (instrument types, families, "
                "qualities, or frequency range)."
            )

    def _prepare_examples(
        self,
        instrument_sources: Optional[List[str]],
        instrument_families: Optional[List[str]],
        qualities: Optional[List[str]],
        fmin: float,
        fmax: float,
    ) -> List[Tuple[str, Dict]]:
        """
        Filters and prepares examples based on specified criteria including pitch frequency range.

        Args:
            instrument_sources: List of instrument source types to include
            instrument_families: List of instrument family types to include
            qualities: List of note qualities to include
            fmin: Minimum frequency in Hz to include
            fmax: Maximum frequency in Hz to include

        Returns:
            List[Tuple[str, Dict]]: List of (note_id, metadata) pairs meeting all criteria
        """
        examples = []
        for note_str, info in self.metadata.items():
            # Convert MIDI pitch to Hz
            pitch_hz = self.midi_to_hz(info["pitch"])
            info["pitch_hz"] = pitch_hz

            # Apply frequency range filter
            if not (fmin <= pitch_hz <= fmax):
                continue

            # Apply other filters
            if (
                instrument_sources
                and info["instrument_source_str"] not in instrument_sources
            ):
                continue
            if (
                instrument_families
                and info["instrument_family_str"] not in instrument_families
            ):
                continue
            if qualities and not any(q in info["qualities_str"] for q in qualities):
                continue

            examples.append((note_str, info))

        return examples

    @staticmethod
    def midi_to_hz(midi_note: int) -> float:
        """
        Converts MIDI note number to frequency in Hz using the standard formula.

        Args:
            midi_note (int): MIDI note number (0-127)

        Returns:
            float: Frequency in Hz

        Note:
            The formula used is: f = 440 * 2^((n-69)/12)
            where f is the frequency in Hz and n is the MIDI note number.
            MIDI note 69 corresponds to A4 (440 Hz).
        """
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    def _detect_voiced_frames(
        self, waveform: torch.Tensor, num_frames: int, frame_length: int = None
    ) -> torch.Tensor:
        """
        Detect voiced frames in the audio by finding portions above the silence threshold.
        Uses vectorized PyTorch operations for faster computation of RMS energy.

        Args:
            waveform (torch.Tensor): Input audio tensor
            num_frames (int): Number of frames to analyze
            frame_length (int, optional): Number of samples per frame. Defaults to hop_size.

        Returns:
            torch.Tensor: Binary tensor with 1s for voiced frames and 0s for silent frames
        """
        if frame_length is None:
            frame_length = self.hop_size

        # Ensure waveform is 1D
        waveform = waveform.squeeze()

        # Pad the waveform to ensure consistent frame sizes
        # This avoids issues with incomplete frames at the end
        total_samples_needed = (num_frames - 1) * self.hop_size + frame_length
        padding_needed = max(0, total_samples_needed - waveform.size(-1))
        if padding_needed > 0:
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed))

        # Use unfold for faster frame extraction (vectorized operation)
        # This creates a tensor of shape [num_frames, frame_length]
        frames = waveform.unfold(0, frame_length, self.hop_size)[:num_frames]

        # Calculate RMS for all frames at once (vectorized)
        # Square all samples, take mean across frame dimension, then sqrt
        rms = torch.sqrt(torch.mean(frames**2, dim=1))

        # Convert to dB scale with safe handling of small values
        eps = 1e-10
        max_rms = torch.max(rms)

        # Avoid division by zero
        if max_rms > eps:
            # Convert to dB: 20 * log10(amplitude / reference)
            rms_db = 20 * torch.log10(torch.clamp(rms / max_rms, min=1e-5))
        else:
            # If max_rms is too small, all frames are likely silent
            rms_db = torch.full_like(rms, -100.0)

        # Find the indices of frames that are NOT silent
        non_silent_frames = (rms_db > self.silence_threshold_db).nonzero(as_tuple=True)[
            0
        ]

        voiced_mask = torch.zeros(num_frames)

        if len(non_silent_frames) > 0:
            # The last non-silent frame index
            last_voiced_frame_index = non_silent_frames[-1]
            voiced_mask[: last_voiced_frame_index + 1] = 1

        return voiced_mask

    def __len__(self) -> int:
        """Returns the number of examples in the filtered dataset."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Path]]:
        """
        Retrieves a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve

        Returns:
            Dict[str, Union[torch.Tensor, Path]]: Dictionary containing:
                - 'audio': Audio waveform tensor [1, samples]
                - 'pitch': Pitch values tensor [frames]
                - 'periodicity': Periodicity values tensor [frames]
                - 'wav_path': Path to the source WAV file

        Raises:
            IndexError: If index is out of range
            IOError: If there are errors loading the audio file
        """
        if not 0 <= idx < len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        note_str, info = self.examples[idx]
        wav_path = self.root_dir / "audio" / f"{note_str}.wav"

        if idx not in self.data_cache or not self.use_cache:
            try:
                waveform, sample_rate = torchaudio.load(wav_path)
            except Exception as e:
                raise IOError(f"Error loading audio file {wav_path}: {str(e)}")

            # Create pitch and periodicity signals
            # NSynth uses constant pitch throughout each note
            num_frames = 1 + (waveform.size(-1) // self.hop_size)
            pitch = torch.full((num_frames,), info["pitch_hz"])

            # Detect voiced frames using energy-based method
            periodicity = self._detect_voiced_frames(waveform, num_frames)

            # Process the sample
            waveform, pitch, periodicity = self.process_sample(
                waveform, pitch, periodicity, sample_rate
            )

            if self.use_cache:
                self.data_cache[idx] = (waveform, pitch, periodicity)
        else:
            waveform, pitch, periodicity = self.data_cache[idx]

        return {
            "audio": waveform,
            "pitch": pitch,
            "periodicity": periodicity,
            "wav_path": wav_path,
        }


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
