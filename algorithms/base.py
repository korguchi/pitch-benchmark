import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class PitchAlgorithm(ABC):
    """Abstract base class for pitch detection algorithms."""

    def __init__(self, sample_rate: int, hop_size: int, fmin: float, fmax: float):
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

    @property
    def supports_continuous_periodicity(self) -> bool:
        """Check if this algorithm supports continuous periodicity extraction."""
        return isinstance(self, ContinuousPitchAlgorithm)

    def _validate_audio(self, audio: np.ndarray) -> None:
        """Validate input audio array."""
        if audio.size == 0:
            raise ValueError("Empty audio input")
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains non-finite values")
        if not (-1.0 <= audio).all() and (audio <= 1.0).all():
            raise ValueError("Audio must be normalized to [-1, 1]")

    def _resample(
        self, pitch: np.ndarray, periodicity: np.ndarray, audio_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align pitch and periodicity to match the audio length based on hop size."""
        target_length = audio_length // self.hop_size
        if target_length <= 0 or len(pitch) == 0:
            return np.zeros(max(0, target_length)), np.zeros(max(0, target_length))

        original_points = np.linspace(0, 1, len(pitch))
        target_points = np.linspace(0, 1, target_length)
        aligned_pitch = np.interp(target_points, original_points, pitch)
        aligned_periodicity = np.interp(target_points, original_points, periodicity)
        return aligned_pitch, aligned_periodicity

    def _sanity_check(
        self, pitch: np.ndarray, periodicity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform sanity checks and cleaning on pitch and periodicity values."""
        periodicity = np.nan_to_num(periodicity, nan=0.0)
        pitch = np.nan_to_num(pitch, nan=0.0)
        pitch = np.clip(pitch, self.fmin, self.fmax)
        periodicity = np.clip(periodicity, 0.0, 1.0)
        return pitch, periodicity

    @abstractmethod
    def extract_pitch(
        self, audio: np.ndarray, threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch and binary voicing from audio.

        Args:
            audio: Input audio signal, normalized to [-1, 1]
            threshold: Voicing threshold. If None, uses algorithm's default.

        Returns:
            Tuple containing:
                - Pitch values (Hz)
                - Binary voicing decisions (0.0 or 1.0)
        """
        pass

    def extract_continuous_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch with continuous periodicity values.

        Args:
            audio: Input audio signal, normalized to [-1, 1]

        Returns:
            Tuple containing:
                - Pitch values (Hz)
                - Continuous periodicity values (0.0 to 1.0)

        Raises:
            NotImplementedError: If algorithm doesn't support continuous periodicity
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support continuous periodicity extraction. "
            f"Use extract_pitch() instead or check supports_continuous_periodicity property."
        )

    @classmethod
    def get_name(cls) -> str:
        """Return the algorithm name."""
        # Use explicit name attribute if available, otherwise derive from class name
        return getattr(cls, "_name", cls.__name__.replace("PitchAlgorithm", ""))


class ContinuousPitchAlgorithm(PitchAlgorithm):
    """Base class for algorithms that can return continuous periodicity."""

    @abstractmethod
    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract raw pitch and continuous periodicity values from audio."""
        pass

    def extract_continuous_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch with continuous periodicity values."""
        self._validate_audio(audio)
        pitch, periodicity = self._extract_raw_pitch_and_periodicity(audio)
        pitch, periodicity = self._sanity_check(pitch, periodicity)
        pitch, periodicity = self._resample(pitch, periodicity, len(audio))
        return pitch, periodicity

    def extract_pitch(
        self, audio: np.ndarray, threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch and binary voicing information."""
        if threshold is None:
            threshold = self._get_default_threshold()

        pitch, periodicity = self.extract_continuous_periodicity(audio)
        voicing = (periodicity >= threshold).astype(bool)
        return pitch, voicing

    def _get_default_threshold(self) -> float:
        """Get the default threshold for this algorithm. Override in subclasses."""
        return 0.5


class ThresholdPitchAlgorithm(PitchAlgorithm):
    """Base class for algorithms that need threshold during extraction."""

    @abstractmethod
    def _extract_pitch_with_threshold(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch and binary periodicity using threshold during extraction."""
        pass

    def _get_default_threshold(self) -> float:
        """Get the default threshold for this algorithm. Override in subclasses."""
        return 0.5

    def extract_pitch(
        self, audio: np.ndarray, threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch and binary voicing information."""
        if threshold is None:
            threshold = self._get_default_threshold()

        self._validate_audio(audio)
        pitch, periodicity = self._extract_pitch_with_threshold(audio, threshold)
        pitch, periodicity = self._sanity_check(pitch, periodicity)
        pitch, periodicity = self._resample(pitch, periodicity, len(audio))
        return pitch, periodicity
