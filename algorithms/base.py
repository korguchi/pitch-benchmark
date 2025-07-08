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
        if not (-1.0 <= audio).all() or not (audio <= 1.0).all():
            raise ValueError("Audio must be normalized to [-1, 1]")

    def _compute_target_times(self, audio_length: int) -> np.ndarray:
        """Generate uniform time grid based on hop_size."""
        n_hops = audio_length // self.hop_size
        return np.arange(n_hops) * (self.hop_size / self.sample_rate)

    def _align_to_grid(
        self, algorithm_times: np.ndarray, values: np.ndarray, target_times: np.ndarray
    ) -> np.ndarray:
        """Align values from algorithm's time grid to target grid."""
        if len(algorithm_times) == 0:
            return np.zeros_like(target_times)

        return np.interp(target_times, algorithm_times, values, left=0.0, right=0.0)

    def _sanity_check(
        self, pitch: np.ndarray, periodicity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sanity checks and cleaning of pitch/periodicity values."""
        periodicity = np.nan_to_num(periodicity, nan=0.0)
        pitch = np.nan_to_num(pitch, nan=0.0)

        # Preserve unvoiced regions (periodicity <= 0)
        voiced = periodicity > 0
        pitch[~voiced] = 0.0
        pitch[voiced] = np.clip(pitch[voiced], self.fmin, self.fmax)

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract raw pitch and continuous periodicity values from audio.

        Returns:
            Tuple containing:
                - Time points (seconds)
                - Pitch values (Hz)
                - Periodicity values (0.0 to 1.0)
        """
        pass

    def extract_continuous_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch with continuous periodicity values."""
        self._validate_audio(audio)
        times, pitch, periodicity = self._extract_raw_pitch_and_periodicity(audio)
        pitch, periodicity = self._sanity_check(pitch, periodicity)

        target_times = self._compute_target_times(len(audio))
        aligned_pitch = self._align_to_grid(times, pitch, target_times)
        aligned_periodicity = self._align_to_grid(times, periodicity, target_times)

        return aligned_pitch, aligned_periodicity

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract pitch and binary periodicity using threshold during extraction.

        Returns:
            Tuple containing:
                - Time points (seconds)
                - Pitch values (Hz)
                - Periodicity values (0.0 or 1.0)
        """
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
        times, pitch, periodicity = self._extract_pitch_with_threshold(audio, threshold)
        pitch, periodicity = self._sanity_check(pitch, periodicity)

        target_times = self._compute_target_times(len(audio))
        aligned_pitch = self._align_to_grid(times, pitch, target_times)
        aligned_periodicity = self._align_to_grid(times, periodicity, target_times)

        return aligned_pitch, aligned_periodicity.astype(bool)
