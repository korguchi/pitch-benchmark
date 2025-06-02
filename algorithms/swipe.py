import numpy as np
from pysptk import sptk
from typing import Tuple
from .base import PitchAlgorithm


class SWIPEPitchAlgorithm(PitchAlgorithm):
    def extract_pitch_and_periodicity(
        self, audio: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch using SWIPE.

        Args:
            audio: Input audio signal
            threshold: Threshold value

        Returns:
            Tuple containing:
                - Pitch frequencies in Hz
                - Binary voicing decisions
        """
        # SWIPE expects a special range.
        # Map threshold from [0,1] to [0.2,0.5]
        norm_threshold = np.clip(0.2 + threshold * (0.5 - 0.2), 0.2, 0.5)

        f0 = sptk.swipe(
            audio,
            self.sample_rate,
            self.hop_size,
            min=self.fmin,
            max=self.fmax,
            threshold=norm_threshold,
            otype="f0",
        )

        return f0, (f0 >= self.fmin).astype(np.float32)
