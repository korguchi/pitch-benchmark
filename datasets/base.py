from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


class PitchDataset(ABC, torch.utils.data.Dataset):
    """
    Abstract base class for audio datasets with pitch and periodicity processing.

    Provides core functionality for audio processing, pitch validation, and resampling
    while defining an interface that derived classes must implement.

    Args:
        sample_rate (int): Target sample rate in Hz
        hop_size (int): Number of audio samples between consecutive frames
        clip_pitch (bool, optional): Whether to clip pitch values to [fmin, fmax] range.
            If False (default), out-of-range pitch values are preserved but their
            periodicity is set to zero, indicating unreliable pitch detection.
            This prevents false pitch information while maintaining data integrity.
            Defaults to False
        normalize_audio (bool, optional): Whether to normalize audio to [-1, 1]. Defaults to True
    """

    DEFAULT_FMIN = 46.875  # Default minimum frequency (G1)
    DEFAULT_FMAX = 2093.75  # Default maximum frequency (C7)

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        clip_pitch: bool = False,
        normalize_audio: bool = True,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_size = hop_size

        # Check if the subclass has defined its own fmin/fmax.
        # If not, getattr falls back to the default values.
        # self.__class__ refers to the specific subclass (e.g., PitchDatasetBach10Synth).
        self.fmin = getattr(self.__class__, "fmin", self.DEFAULT_FMIN)
        self.fmax = getattr(self.__class__, "fmax", self.DEFAULT_FMAX)

        self.clip_pitch = clip_pitch
        self.normalize_audio = normalize_audio
        self._validate_init_params(sample_rate, hop_size, self.fmin, self.fmax)

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

    def get_group(self, idx: int) -> str:
        """Return group identifier for sample (speaker/instrument)"""
        return str(idx)  # Default: each sample is its own group

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
            raise ValueError("Silent audio!")

        if self.normalize_audio:
            max_abs = audio.abs().max()
            if max_abs > 1:  # Normalize only if the range exceeds -1 to 1
                audio = audio / max_abs

        return audio.clamp(-1.0, 1.0)

    def _validate_pitch(
        self,
        pitch: torch.Tensor,
        periodicity: torch.Tensor,
        notes: Optional[List[Dict]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Dict]]]:
        """
        Validates and processes pitch, periodicity values, and optionally notes.

        By default, pitch values outside the [fmin, fmax] range are preserved
        but their corresponding periodicity is set to zero. This approach maintains
        data integrity while indicating that pitch detection is unreliable outside
        the specified frequency range, avoiding false pitch information.

        The same frequency constraints are applied to transcription notes to ensure
        consistency with the pitch algorithm's operational range.

        Args:
            pitch (torch.Tensor): Pitch values
            periodicity (torch.Tensor): Periodicity values
            notes (Optional[List[Dict]]): Musical notes with 'start', 'end', 'midi_pitch'

        Returns:
            Tuple containing:
            - pitch (torch.Tensor): Processed pitch values
            - periodicity (torch.Tensor): Processed periodicity values
            - notes (Optional[List[Dict]]): Processed notes (if provided)

            If clip_pitch=True, pitch values are clipped to [fmin, fmax] range and
            note frequencies are similarly clipped.
            If clip_pitch=False, out-of-range pitch values have periodicity set to 0
            and out-of-range notes are excluded.
        """
        # Validate pitch and periodicity shapes
        if pitch.shape != periodicity.shape:
            raise ValueError(
                f"Pitch and periodicity shapes must match: {pitch.shape} vs {periodicity.shape}"
            )

        # Clean up pitch values
        pitch = torch.nan_to_num(pitch, nan=self.fmin)

        # Ensure periodicity is binary
        periodicity = torch.round(periodicity).clamp(0, 1)

        if self.clip_pitch:
            # Clip pitch values to valid range
            pitch = torch.clamp(pitch, self.fmin, self.fmax)

            # Apply same clipping logic to notes if provided
            if notes is not None:
                processed_notes = []
                for note in notes:
                    midi_pitch = note["midi_pitch"]
                    freq_hz = 440.0 * (2 ** ((midi_pitch - 69) / 12))

                    # Clip frequency to algorithm's operational range
                    freq_hz_clipped = max(self.fmin, min(freq_hz, self.fmax))
                    midi_pitch_clipped = 69 + 12 * np.log2(freq_hz_clipped / 440.0)

                    processed_note = note.copy()
                    processed_note["midi_pitch"] = float(midi_pitch_clipped)
                    processed_notes.append(processed_note)
                notes = processed_notes
        else:
            # Preserve pitch but zero periodicity for out-of-range values
            out_of_range_mask = (pitch < self.fmin) | (pitch > self.fmax)
            periodicity = periodicity * (~out_of_range_mask).float()

            # Apply same filtering logic to notes if provided
            if notes is not None:
                processed_notes = []
                for note in notes:
                    midi_pitch = note["midi_pitch"]
                    freq_hz = 440.0 * (2 ** ((midi_pitch - 69) / 12))

                    # Only include notes within algorithm's operational range
                    if self.fmin <= freq_hz <= self.fmax:
                        processed_notes.append(note.copy())
                    # Notes outside range are excluded (algorithm can't detect them)
                notes = processed_notes

        return pitch, periodicity.bool(), notes

    def process_sample(
        self,
        audio: torch.Tensor,
        pitch: torch.Tensor,
        periodicity: torch.Tensor,
        orig_sr: int,
        notes: Optional[List[Dict]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[Dict]]]:
        """
        Processes a single audio sample with its corresponding pitch, periodicity, and (optionally) notes.

        Applies consistent frequency constraints to both F0 estimates and transcription notes to ensure
        they represent what the pitch algorithm can actually detect.

        Args:
            audio (torch.Tensor): Audio waveform
            pitch (torch.Tensor): Pitch values
            periodicity (torch.Tensor): Periodicity values
            orig_sr (int): Original sample rate of the audio
            notes (Optional[List[Dict]]): Musical notes with 'start', 'end', 'midi_pitch'

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[Dict]]]: Processed audio, pitch, periodicity and notes
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
        target_length = audio.size(-1) // self.hop_size

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
        pitch, periodicity, notes = self._validate_pitch(pitch, periodicity, notes)

        if notes is None:
            return audio.squeeze(0), pitch, periodicity
        else:
            return audio.squeeze(0), pitch, periodicity, notes

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Path]]:
        """Retrieves a single item from the dataset."""
        pass
