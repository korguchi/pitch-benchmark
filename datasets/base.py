import torch
from pathlib import Path
import torchaudio
import torch.nn.functional as F
from typing import Dict, Tuple, Union
from abc import ABC, abstractmethod


class PitchDataset(ABC, torch.utils.data.Dataset):
    """
    Abstract base class for audio datasets with pitch and periodicity processing.

    Provides core functionality for audio processing, pitch validation, and resampling
    while defining an interface that derived classes must implement.

    Args:
        sample_rate (int): Target sample rate in Hz
        hop_size (int): Number of audio samples between consecutive frames
        fmin (float, optional): Minimum frequency in Hz. Defaults to 20.0
        fmax (float, optional): Maximum frequency in Hz. Defaults to 2000.0
        clip_pitch (bool, optional): Whether to clip pitch values to [fmin, fmax] range.
            If False (default), out-of-range pitch values are preserved but their
            periodicity is set to zero, indicating unreliable pitch detection.
            This prevents false pitch information while maintaining data integrity.
            Defaults to False
        normalize_audio (bool, optional): Whether to normalize audio to [-1, 1]. Defaults to True
    """

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        fmin: float = 20.0,
        fmax: float = 2000.0,
        clip_pitch: bool = False,
        normalize_audio: bool = True,
    ):
        super().__init__()
        self._validate_init_params(sample_rate, hop_size, fmin, fmax)

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.clip_pitch = clip_pitch
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
            raise ValueError("Silent audio!")

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

        By default, pitch values outside the [fmin, fmax] range are preserved
        but their corresponding periodicity is set to zero. This approach maintains
        data integrity while indicating that pitch detection is unreliable outside
        the specified frequency range, avoiding false pitch information.

        Args:
            pitch (torch.Tensor): Pitch values
            periodicity (torch.Tensor): Periodicity values

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Processed pitch and periodicity values.
                If clip_pitch=True, pitch values are clipped to [fmin, fmax] range.
                If clip_pitch=False, out-of-range pitch values have periodicity set to 0.
        """
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
        else:
            # Preserve pitch but zero periodicity for out-of-range values
            out_of_range_mask = (pitch < self.fmin) | (pitch > self.fmax)
            periodicity = periodicity * (~out_of_range_mask).float()

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
