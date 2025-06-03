import numpy as np
import crepe
import tensorflow as tf
from typing import Tuple
from .base import ContinuousPitchAlgorithm


class CREPEPitchAlgorithm(ContinuousPitchAlgorithm):
    def __init__(
        self,
        viterbi: bool = True,
        model: str = "full",
        device: str = None,
        **kwargs,
    ):
        """Initialize CREPE pitch detector.
        Args:
            viterbi: Whether to use Viterbi decoding
            model: Model capacity ('tiny', 'small', 'medium', 'large', or 'full')
            device: Device to use ('cpu' or 'cuda')
        """
        super().__init__(**kwargs)
        self.viterbi = viterbi
        self.model = model
        self.step_size = (self.hop_size / self.sample_rate) * 1000

        # Set up TensorFlow device
        if device is None:
            self.tf_device = (
                "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
            )
        elif device == "cuda":
            if not tf.config.list_physical_devices("GPU"):
                print(
                    "Warning: CUDA requested but no GPU available. Falling back to CPU."
                )
                self.tf_device = "/CPU:0"
            else:
                self.tf_device = "/GPU:0"
        else:  # device == 'cpu'
            self.tf_device = "/CPU:0"

        # Force TensorFlow to only use the specified device
        if self.tf_device == "/CPU:0":
            tf.config.set_visible_devices([], "GPU")

    def _extract_raw_pitch_and_periodicity(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        with tf.device(self.tf_device):
            _, frequency, confidence, _ = crepe.predict(
                audio,
                self.sample_rate,
                model_capacity=self.model,
                viterbi=self.viterbi,
                step_size=self.step_size,
                verbose=0,
            )
        return frequency, confidence
