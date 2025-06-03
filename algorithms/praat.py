import numpy as np
import parselmouth
from typing import Tuple
from .base import ContinuousPitchAlgorithm


class PraatPitchAlgorithm(ContinuousPitchAlgorithm):
    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        sound = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
        time_step = self.hop_size / self.sample_rate

        pitch_obj = sound.to_pitch(
            time_step=time_step, pitch_floor=self.fmin, pitch_ceiling=self.fmax
        )

        pitch_values = pitch_obj.selected_array["frequency"]
        strength_values = pitch_obj.selected_array["strength"]

        if len(pitch_values) > 0:
            valid_mask = strength_values > 0
            if valid_mask.any():
                valid_indices = np.where(valid_mask)[0]
                pitch_values = np.interp(
                    np.arange(len(pitch_values)),
                    valid_indices,
                    pitch_values[valid_indices],
                )

        return pitch_values, strength_values
