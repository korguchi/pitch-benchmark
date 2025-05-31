import torch
import json
from pathlib import Path
import torchaudio
from typing import Dict, List, Tuple, Optional, Union
from .base import PitchDataset


class PitchDatasetNSynth(PitchDataset):
    """
    Dataset implementation for the NSynth (Neural Audio Synthesis) dataset.

    This class handles loading and processing of the NSynth dataset, which contains
    musical notes from various instruments. It provides filtering capabilities based
    on instrument types, families and note qualities.

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
        silence_threshold_db: float = -40.0,
        **kwargs,
    ):
        # Initialize parent class
        super().__init__(**kwargs)

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
            instrument_sources, instrument_families, qualities
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
    ) -> List[Tuple[str, Dict]]:
        """
        Filters and prepares examples based on specified criteria including instrument sources.

        Args:
            instrument_sources: List of instrument source types to include
            instrument_families: List of instrument family types to include
            qualities: List of note qualities to include

        Returns:
            List[Tuple[str, Dict]]: List of (note_id, metadata) pairs meeting all criteria
        """
        examples = []
        for note_str, info in self.metadata.items():
            # Convert MIDI pitch to Hz
            pitch_hz = self.midi_to_hz(info["pitch"])
            info["pitch_hz"] = pitch_hz

            # Apply filters
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
