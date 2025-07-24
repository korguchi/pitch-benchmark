import numpy as np
from typing import Tuple, Optional, List, Dict
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

    def notes_from_pitch_contour(
        self,
        pitch_contour: np.ndarray,
        voicing_contour: np.ndarray,
        split_semitone_threshold: float = 0.8,
        min_note_duration: float = 0.05,
        unvoiced_grace_period: float = 0.02,
    ) -> List[Dict[str, float]]:
        """
        Segments a pitch contour into discrete musical notes.

        Splits the contour when pitch deviates by more than split_semitone_threshold
        semitones from the current note's median pitch. Includes a grace period for
        unvoiced frames to avoid premature note termination.

        Args:
            pitch_contour: Array of fundamental frequencies in Hz
            voicing_contour: Array of voicing probabilities (0 to 1)
            split_semitone_threshold: Pitch difference in semitones to trigger a new
                note segment. Recommended range: 0.6-1.2
            min_note_duration: Minimum duration in seconds for a valid note
            unvoiced_grace_period: Maximum duration in seconds of unvoiced segments
                that are still considered part of the current note

        Returns:
            List of note dictionaries with 'start', 'end', and 'midi_pitch' keys
        """
        frame_period = self.hop_size / self.sample_rate
        notes = []
        current_note_segment = None
        unvoiced_frames_count = 0

        # Pre-compute valid voiced frames mask
        valid_voiced_frames = (
            (voicing_contour > 0)
            & (pitch_contour >= self.fmin)
            & (pitch_contour <= self.fmax)
        )

        # Convert valid pitch values to MIDI semitones (vectorized operation)
        midi_contour = np.full_like(pitch_contour, np.nan)
        valid_indices = np.where(valid_voiced_frames)[0]
        if len(valid_indices) > 0:
            midi_contour[valid_indices] = 69 + 12 * np.log2(
                pitch_contour[valid_indices] / 440.0
            )

        for i, is_voiced in enumerate(valid_voiced_frames):
            t = i * frame_period

            if is_voiced:
                unvoiced_frames_count = 0
                midi_pitch = midi_contour[i]

                if current_note_segment is None:
                    # Start new note segment
                    current_note_segment = {
                        "start": t,
                        "end": t + frame_period,
                        "samples": [midi_pitch],
                    }
                else:
                    # Check if pitch deviation exceeds threshold
                    current_median = np.median(current_note_segment["samples"])
                    pitch_deviation = abs(midi_pitch - current_median)

                    if pitch_deviation >= split_semitone_threshold:
                        # Finalize current note and start new one
                        notes.append(current_note_segment)
                        current_note_segment = {
                            "start": t,
                            "end": t + frame_period,
                            "samples": [midi_pitch],
                        }
                    else:
                        # Continue current note
                        current_note_segment["samples"].append(midi_pitch)
                        current_note_segment["end"] = t + frame_period

            else:  # Unvoiced frame
                if current_note_segment is not None:
                    unvoiced_frames_count += 1
                    unvoiced_duration = unvoiced_frames_count * frame_period

                    if unvoiced_duration >= unvoiced_grace_period:
                        # Grace period exceeded - finalize current note
                        notes.append(current_note_segment)
                        current_note_segment = None
                    else:
                        # Within grace period - extend note duration
                        current_note_segment["end"] = t + frame_period

        # Finalize last note segment if it exists
        if current_note_segment is not None:
            notes.append(current_note_segment)

        if not notes:
            return []

        # Filter by duration and compute final MIDI pitches
        processed_notes = []
        for segment in notes:
            duration = segment["end"] - segment["start"]
            if duration >= min_note_duration and segment["samples"]:
                median_pitch = np.median(segment["samples"])
                processed_notes.append(
                    {
                        "start": segment["start"],
                        "end": segment["end"],
                        "midi_pitch": round(median_pitch),  # int() call is redundant
                    }
                )

        if not processed_notes:
            return []

        # Merge adjacent notes with identical MIDI pitch
        final_notes = [processed_notes[0]]
        epsilon = 1e-9  # For floating-point precision

        for current_note in processed_notes[1:]:
            previous_note = final_notes[-1]
            gap = current_note["start"] - previous_note["end"]

            # Merge if notes are adjacent and have same pitch
            if (
                gap <= frame_period + epsilon
                and previous_note["midi_pitch"] == current_note["midi_pitch"]
            ):
                previous_note["end"] = current_note["end"]
            else:
                final_notes.append(current_note)

        return final_notes

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
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
        """Extract pitch and binary voicing information."""
        if threshold is None:
            threshold = self._get_default_threshold()

        pitch, periodicity = self.extract_continuous_periodicity(audio)
        voicing = (periodicity >= threshold).astype(bool)
        notes = self.notes_from_pitch_contour(pitch, voicing)

        return pitch, voicing, notes

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
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
        """Extract pitch and binary voicing information."""
        if threshold is None:
            threshold = self._get_default_threshold()

        self._validate_audio(audio)
        times, pitch, periodicity = self._extract_pitch_with_threshold(audio, threshold)
        pitch, periodicity = self._sanity_check(pitch, periodicity)

        target_times = self._compute_target_times(len(audio))
        aligned_pitch = self._align_to_grid(times, pitch, target_times)
        aligned_periodicity = self._align_to_grid(times, periodicity, target_times)
        notes = self.notes_from_pitch_contour(
            aligned_pitch, aligned_periodicity.astype(bool)
        )

        return aligned_pitch, aligned_periodicity.astype(bool), notes
