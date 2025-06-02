import numpy as np
import librosa
from typing import Tuple
from .base import PitchAlgorithm


class pYINPitchAlgorithm(PitchAlgorithm):
    def extract_pitch_and_periodicity(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch using pYIN.

        Args:
            audio: Input audio signal
            threshold: Not used (pYIN has its own probability measure)

        Returns:
            Tuple containing:
                - Pitch frequencies in Hz
                - Combined voicing/probability flags
        """
        pitch, voiced_flag, prob_flag = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sample_rate,
            hop_length=self.hop_size,
        )

        return pitch, np.maximum(voiced_flag, prob_flag)
