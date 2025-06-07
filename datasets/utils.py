import torch
import torch.nn.functional as F
from typing import Dict, Union, List, Tuple
from pathlib import Path

from .base import PitchDataset


class ChunkedPitchDataset:
    """
    Dataset wrapper that extracts fixed-duration chunks from PitchDataset instances.

    This wrapper processes audio files by dividing them into non-overlapping chunks of
    specified duration. Each chunk maintains frame alignment between audio samples and
    pitch/periodicity annotations, ensuring consistent input-output relationships for
    training models.

    The chunking process:
    1. Divides each audio file into non-overlapping segments of `chunk_duration` seconds
    2. Maintains frame alignment: chunk_samples = chunk_duration * sample_rate
    3. Expected output frames = chunk_samples // hop_size
    4. Pads shorter chunks and annotations to maintain consistent tensor sizes
    5. Filters out chunks too short to contain meaningful pitch information

    Args:
        dataset (PitchDataset): Underlying dataset instance that provides audio,
            pitch, and periodicity data
        chunk_duration (float, optional): Duration of each chunk in seconds.
            Defaults to 2.0 seconds
        pad_value (float, optional): Value to use for padding both pitch and
            periodicity tensors when chunks are shorter than expected.
            Defaults to NaN

    Example:
        >>> base_dataset = MyPitchDataset(sample_rate=22050, hop_size=256)
        >>> chunked_dataset = ChunkedPitchDataset(base_dataset, chunk_duration=1.5)
        >>> print(f"Original dataset: {len(base_dataset)} files")
        >>> print(f"Chunked dataset: {len(chunked_dataset)} chunks")
        >>> sample = chunked_dataset[0]
        >>> print(f"Audio shape: {sample['audio'].shape}")
        >>> print(f"Pitch shape: {sample['pitch'].shape}")
    """

    def __init__(
        self,
        dataset: PitchDataset,
        chunk_duration: float = 2.0,
        pad_value: float = float("nan"),
    ):
        if not isinstance(dataset, PitchDataset):
            raise TypeError("Dataset must be an instance of PitchDataset")
        if chunk_duration <= 0:
            raise ValueError(f"chunk_duration must be positive, got {chunk_duration}")

        self.dataset = dataset
        self.chunk_duration = chunk_duration
        self.pad_value = pad_value

        # Compute chunk parameters
        self.chunk_samples = int(chunk_duration * self.dataset.sample_rate)
        self.expected_frames = self.chunk_samples // self.dataset.hop_size

        # Precompute all valid chunk positions
        self._compute_chunk_indices()

    def _compute_chunk_indices(self) -> None:
        """
        Precomputes valid chunk positions for all files in the dataset.

        Creates a list of (file_index, audio_start_sample) tuples representing
        all valid chunks across the dataset. Chunks must contain at least
        one hop_size worth of samples to be included.
        """
        self.chunks: List[Tuple[int, int]] = []

        for file_idx in range(len(self.dataset)):
            try:
                sample = self.dataset[file_idx]
                audio = sample["audio"]
                total_samples = audio.size(-1)

                # Generate non-overlapping chunks
                audio_start = 0
                while audio_start < total_samples:
                    audio_end = min(audio_start + self.chunk_samples, total_samples)
                    chunk_length = audio_end - audio_start

                    # Only include chunks with at least one hop_size worth of samples
                    if chunk_length >= self.dataset.hop_size:
                        self.chunks.append((file_idx, audio_start))

                    audio_start += self.chunk_samples

            except Exception as e:
                print(f"Warning: Skipping file {file_idx} due to error: {e}")
                continue

    def __len__(self) -> int:
        """Returns the total number of chunks across all files."""
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Path, int]]:
        """
        Retrieves a single chunk from the dataset.

        Args:
            idx (int): Chunk index

        Returns:
            Dict containing:
                - All original sample data from the base dataset
                - 'audio': Chunked audio tensor, padded to chunk_samples length
                - 'pitch': Chunked pitch tensor, padded to expected_frames length
                - 'periodicity': Chunked periodicity tensor, padded to expected_frames length
                - 'chunk_start': Audio sample index where chunk begins
                - 'chunk_end': Audio sample index where chunk ends
                - 'file_idx': Index of the original file in the base dataset

        Raises:
            IndexError: If idx is out of range
        """
        if not (0 <= idx < len(self.chunks)):
            raise IndexError(f"Index {idx} out of range [0, {len(self.chunks)})")

        file_idx, audio_start = self.chunks[idx]
        sample = self.dataset[file_idx]

        audio = sample["audio"]
        pitch = sample["pitch"]
        periodicity = sample["periodicity"]
        total_samples = audio.size(-1)

        # Extract audio chunk
        audio_end = min(audio_start + self.chunk_samples, total_samples)
        audio_chunk = audio[..., audio_start:audio_end]

        # Pad audio to expected length if needed
        current_length = audio_chunk.size(-1)
        if current_length < self.chunk_samples:
            pad_length = self.chunk_samples - current_length
            audio_chunk = F.pad(
                audio_chunk, (0, pad_length), mode="constant", value=0.0
            )

        # Calculate frame indices for annotations
        frame_start = audio_start // self.dataset.hop_size
        frame_end = frame_start + self.expected_frames

        # Extract annotation chunks
        pitch_chunk = pitch[..., frame_start:frame_end]
        periodicity_chunk = periodicity[..., frame_start:frame_end]

        # Pad annotations to expected frame count if needed
        current_frames = pitch_chunk.size(-1)
        if current_frames < self.expected_frames:
            pad_frames = self.expected_frames - current_frames
            pitch_chunk = F.pad(
                pitch_chunk, (0, pad_frames), mode="constant", value=self.pad_value
            )
            periodicity_chunk = F.pad(
                periodicity_chunk,
                (0, pad_frames),
                mode="constant",
                value=self.pad_value,
            )

        # Create output sample
        result = {
            **sample,  # Include all original data
            "audio": audio_chunk,
            "pitch": pitch_chunk,
            "periodicity": periodicity_chunk,
            "chunk_start": audio_start,
            "chunk_end": audio_end,
            "file_idx": file_idx,
        }

        return result

    def __getattr__(self, name: str):
        """
        Delegates attribute access to the underlying dataset.

        This allows the chunked dataset to behave like the original dataset
        for accessing properties like sample_rate, hop_size, etc.

        Args:
            name (str): Attribute name

        Returns:
            The attribute value from the underlying dataset

        Raises:
            AttributeError: If the attribute doesn't exist in the underlying dataset
        """
        return getattr(self.dataset, name)


def pitch_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching pitch dataset samples with dynamic padding.

    This function handles samples with varying lengths by padding them to the maximum
    length in the batch. Audio is zero-padded while pitch and periodicity annotations
    are NaN-padded to indicate missing/invalid data.

    Args:
        batch (List[Dict[str, torch.Tensor]]): List of samples from PitchDataset,
            where each sample contains at least 'audio', 'pitch', and 'periodicity' tensors

    Returns:
        Dict[str, torch.Tensor]: Batched tensors with consistent shapes:
            - 'audio': [batch_size, max_audio_length] - zero-padded
            - 'pitch': [batch_size, max_pitch_length] - NaN-padded
            - 'periodicity': [batch_size, max_pitch_length] - NaN-padded

    Raises:
        KeyError: If required keys ('audio', 'pitch', 'periodicity') are missing
        ValueError: If batch is empty

    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataset = ChunkedPitchDataset(base_dataset)
        >>> loader = DataLoader(dataset, batch_size=4, collate_fn=pitch_collate_fn)
        >>> for batch in loader:
        ...     print(f"Audio shape: {batch['audio'].shape}")
        ...     print(f"Pitch shape: {batch['pitch'].shape}")
    """
    if not batch:
        raise ValueError("Cannot collate empty batch")

    # Validate that all required keys are present
    required_keys = {"audio", "pitch", "periodicity"}
    missing_keys = required_keys - set(batch[0].keys())
    if missing_keys:
        raise KeyError(f"Missing required keys in batch samples: {missing_keys}")

    # Find maximum lengths
    max_audio_length = max(sample["audio"].size(-1) for sample in batch)
    max_pitch_length = max(sample["pitch"].size(-1) for sample in batch)

    # Initialize lists for batched tensors
    audio_list = []
    pitch_list = []
    periodicity_list = []

    for sample in batch:
        audio = sample["audio"]
        pitch = sample["pitch"]
        periodicity = sample["periodicity"]

        # Pad or truncate audio to max length (zero padding)
        if audio.size(-1) != max_audio_length:
            if audio.size(-1) < max_audio_length:
                # Zero pad audio
                audio = torch.nn.functional.pad(
                    audio,
                    (0, max_audio_length - audio.size(-1)),
                    mode="constant",
                    value=0.0,
                )
            else:
                # Truncate audio
                audio = audio[..., :max_audio_length]

        # Pad or truncate pitch to max length (NaN padding)
        if pitch.size(-1) != max_pitch_length:
            if pitch.size(-1) < max_pitch_length:
                # NaN pad pitch
                pitch = torch.nn.functional.pad(
                    pitch,
                    (0, max_pitch_length - pitch.size(-1)),
                    mode="constant",
                    value=float("nan"),
                )
            else:
                # Truncate pitch
                pitch = pitch[..., :max_pitch_length]

        # Pad or truncate periodicity to max length (NaN padding)
        if periodicity.size(-1) != max_pitch_length:
            if periodicity.size(-1) < max_pitch_length:
                # NaN pad periodicity
                periodicity = torch.nn.functional.pad(
                    periodicity,
                    (0, max_pitch_length - periodicity.size(-1)),
                    mode="constant",
                    value=float("nan"),
                )
            else:
                # Truncate periodicity
                periodicity = periodicity[..., :max_pitch_length]

        audio_list.append(audio)
        pitch_list.append(pitch)
        periodicity_list.append(periodicity)

    # Stack tensors into batches
    try:
        batched_audio = torch.stack(audio_list, dim=0)
        batched_pitch = torch.stack(pitch_list, dim=0)
        batched_periodicity = torch.stack(periodicity_list, dim=0)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to stack tensors into batch. "
            f"Ensure all samples have compatible shapes: {e}"
        )

    return {
        "audio": batched_audio,
        "pitch": batched_pitch,
        "periodicity": batched_periodicity,
    }
