import numpy as np
import librosa
from typing import Tuple
from .base import ContinuousPitchAlgorithm


class pYINPitchAlgorithm(ContinuousPitchAlgorithm):
    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pitch, voiced_flag, prob_flag = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sample_rate,
            hop_length=self.hop_size,
        )

        return pitch, np.maximum(voiced_flag, prob_flag)
