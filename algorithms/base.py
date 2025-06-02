import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class PitchAlgorithm(ABC):
    """Abstract base class for pitch detection algorithms.

    Implements common functionality for pitch extraction including:
    - Input validation and preprocessing
    - Output resampling and alignment
    - Sanity checks and post-processing
    """

    def __init__(self, sample_rate: int, hop_size: int, fmin: float, fmax: float):
        """Initialize pitch algorithm with required parameters.

        Args:
            sample_rate: Audio sampling rate in Hz
            hop_size: Number of samples between successive frames
            fmin: Minimum detectable frequency in Hz
            fmax: Maximum detectable frequency in Hz

        Raises:
            ValueError: If parameters are invalid
        """
        if fmin >= fmax:
            raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        if hop_size <= 0:
            raise ValueError(f"Hop size must be positive, got {hop_size}")

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax

    def _validate_audio(self, audio: np.ndarray) -> None:
        """Validate input audio array.

        Args:
            audio: Input audio signal

        Raises:
            ValueError: If audio is invalid or empty
        """
        if audio.size == 0:
            raise ValueError("Empty audio input")
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains non-finite values")
        if not (-1.0 <= audio).all() and (audio <= 1.0).all():
            raise ValueError("Audio must be normalized to [-1, 1]")

    def resample(
        self, pitch: np.ndarray, periodicity: np.ndarray, audio_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align pitch and periodicity to match the audio length based on hop size.

        Args:
            pitch: Pitch values of shape (frames,)
            periodicity: Periodicity values of shape (frames,)
            audio_length: Length of the input audio in samples

        Returns:
            Tuple containing:
                - Aligned pitch values
                - Aligned periodicity values
        """
        target_length = audio_length // self.hop_size

        if target_length <= 0 or len(pitch) == 0:
            return np.zeros(max(0, target_length)), np.zeros(max(0, target_length))

        original_points = np.linspace(0, 1, len(pitch))
        target_points = np.linspace(0, 1, target_length)

        aligned_pitch = np.interp(target_points, original_points, pitch)
        aligned_periodicity = np.interp(target_points, original_points, periodicity)

        return aligned_pitch, aligned_periodicity

    def sanity_check(
        self, pitch: np.ndarray, periodicity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform sanity checks and cleaning on pitch and periodicity values.

        Args:
            pitch: Array of pitch values
            periodicity: Array of periodicity/confidence values

        Returns:
            Tuple containing sanitized pitch and periodicity arrays
        """
        periodicity = np.nan_to_num(periodicity, nan=0.0)
        pitch = np.nan_to_num(pitch, nan=0.0)
        pitch = np.clip(pitch, self.fmin, self.fmax)
        periodicity = np.clip(periodicity, 0.0, 1.0)

        return pitch, periodicity

    @abstractmethod
    def extract_pitch_and_periodicity(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract raw pitch and periodicity values from audio.

        Args:
            audio: Input audio array
            threshold: Voicing decision threshold

        Returns:
            Tuple containing:
                - Pitch values in Hz
                - Periodicity/confidence values
        """
        pass

    def __call__(
        self, audio: np.ndarray, threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process audio to extract pitch and voicing information.

        The function handles pitch extraction, periodicity calculation, and resampling
        if needed. The output length is standardized to 1 + audio.size(-1) // hop_size
        to match ground truth pitches, accounting for centering effects.

        Args:
            audio: Input audio signal, normalized to [-1, 1]
            threshold: Voicing decision threshold

        Returns:
            Tuple containing:
                - Pitch values (Hz) with unvoiced frames set to 0
                - Binary voicing decisions

        Raises:
            ValueError: If input audio is invalid

        Notes:
            Resampling is only applied for algorithms that don't naturally output
            the expected number of samples (e.g., Praat) or use different internal
            sampling rates (e.g., CREPE at 16kHz/160 hop). Algorithms with default
            centering like pYIN typically don't require resampling.
        """
        self._validate_audio(audio)
        pitch, periodicity = self.extract_pitch_and_periodicity(audio, threshold)
        pitch, periodicity = self.sanity_check(pitch, periodicity)
        pitch, periodicity = self.resample(pitch, periodicity, len(audio))
        voicing = periodicity >= threshold
        return pitch * voicing, voicing
