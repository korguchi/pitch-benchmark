import numpy as np
from pysptk import sptk
from typing import Tuple
from .base import PitchAlgorithm


class RAPTPitchAlgorithm(PitchAlgorithm):
    def extract_pitch_and_periodicity(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch using RAPT.

        Args:
            audio: Input audio signal
            threshold: Voice bias parameter

        Returns:
            Tuple containing:
                - Pitch frequencies in Hz
                - Binary voicing decisions
        """
        audio_scaled = np.clip(audio * 32767, -32768, 32767)
        # RAPT expects a special range.
        # Map threshold from [0,1] to [-0.6,0.7]
        norm_threshold = -0.6 + threshold * (0.7 - (-0.6))

        f0 = sptk.rapt(
            audio_scaled,
            self.sample_rate,
            self.hop_size,
            min=self.fmin,
            max=self.fmax,
            voice_bias=norm_threshold,
            otype="f0",
        )

        return f0, (f0 >= self.fmin).astype(np.float32)
