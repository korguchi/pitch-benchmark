import torch
import random
from pathlib import Path
from typing import Dict, Tuple, Union, List
from .base import PitchDataset

from typing import Optional

import math
from torch import nn, Tensor


class LayerNorm1d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ConvSeparable(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0,
    ):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding="same",
            groups=in_channels,
            bias=False,
        )
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1)

        std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * out_channels))
        nn.init.normal_(self.depthwise_conv.weight, mean=0, std=std)
        nn.init.normal_(self.pointwise_conv.weight, mean=0, std=std)
        nn.init.zeros_(self.pointwise_conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class SepConvLayer(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.layer_norm = LayerNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation_fn = nn.ReLU(inplace=True)
        self.conv1 = ConvSeparable(channels, channels, kernel_size, dropout=dropout)
        self.conv2 = ConvSeparable(channels, channels, kernel_size, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.activation_fn(self.conv1(x))
        x = self.dropout(x)
        x = self.activation_fn(self.conv2(x))
        x = self.dropout(x)
        return residual + x


class LightSpeech(nn.Module):
    def __init__(
        self,
        num_phones: int,
        num_speakers: int,
        num_mel_bins: int,
        num_tones: int = 7,
        tone_embedding: int = 16,
        d_model: int = 512,
        layer_dropout: float = 0.2,
        encoder_kernel_sizes: List[int] = [5, 25, 13, 9],
        decoder_kernel_sizes: List[int] = [17, 21, 9, 3],
        duration_layers: int = 1,
        duration_kernel_size: int = 3,
        duration_dropout: float = 0.25,
        pitch_layers: int = 6,
        pitch_kernel_size: int = 5,
        pitch_dropout: float = 0.25,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.d_model = d_model

        self.num_speakers = num_speakers
        if self.num_speakers > 1:
            self.speaker_embedding = nn.Embedding(self.num_speakers, d_model)
        self.embed_tokens = nn.Embedding(
            num_phones, d_model - tone_embedding, padding_idx=self.padding_idx
        )
        self.embed_tones = nn.Embedding(
            num_tones, tone_embedding, padding_idx=self.padding_idx
        )
        self.dropout = nn.Dropout(layer_dropout)
        self.embed_pitch = nn.Conv1d(2, d_model, kernel_size=1)

        self.encoder = nn.ModuleList(
            [
                SepConvLayer(d_model, kernel_size, layer_dropout)
                for kernel_size in encoder_kernel_sizes
            ]
        )
        self.decoder = nn.ModuleList(
            [
                SepConvLayer(d_model, kernel_size, layer_dropout)
                for kernel_size in decoder_kernel_sizes
            ]
        )

        self.duration_predictor = self._make_predictor(
            hidden_size=d_model,
            out_dim=1,
            num_layers=duration_layers,
            kernel_size=duration_kernel_size,
            dropout=duration_dropout,
        )
        self.pitch_predictor = self._make_predictor(
            hidden_size=d_model,
            out_dim=2,
            num_layers=pitch_layers,
            kernel_size=pitch_kernel_size,
            dropout=pitch_dropout,
        )

        self.layer_norm = LayerNorm1d(d_model)
        self.layer_norm2 = LayerNorm1d(d_model)
        self.mel_out = nn.Conv1d(d_model, num_mel_bins, kernel_size=1)

    @staticmethod
    def _make_predictor(
        hidden_size: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.5,
        kernel_size: int = 3,
    ):
        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    ConvSeparable(hidden_size, hidden_size, kernel_size),
                    nn.ReLU(inplace=True),
                    LayerNorm1d(hidden_size),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Conv1d(hidden_size, out_dim, kernel_size=1))
        return nn.Sequential(*layers)

    def _length_regulator(self, x: Tensor, mel_time: int, durations: Tensor) -> Tensor:
        bsz, time, feats = x.shape
        if bsz > 1:
            cumulative_durations = torch.cumsum(durations, dim=1)

            # Create a range tensor for each batch item
            expanded_range = (
                torch.arange(mel_time, device=x.device).unsqueeze(0).expand(bsz, -1)
            )

            # Create a mask for valid positions
            mask = expanded_range.unsqueeze(1) >= cumulative_durations.unsqueeze(2)

            # Calculate source indices
            source_indices = mask.long().sum(dim=1)

            # Clamp the indices to handle cases where mel_time > total_duration
            source_indices = torch.clamp(source_indices, 0, time - 1)

            # Create the gather indices tensor
            gather_indices = source_indices.unsqueeze(-1).expand(-1, -1, feats)

            # Gather the input tensor based on the calculated indices
            return torch.gather(x, 1, gather_indices)
        else:
            indices = torch.arange(time, device=x.device)
            repeated_indices = torch.repeat_interleave(
                indices, durations[0].long(), dim=0
            )
            return x[:, repeated_indices]

    def forward(
        self,
        speakers: Tensor,
        tokens: Tensor,
        tones: Tensor,
        pitches: Optional[Tensor] = None,
        periodicity: Optional[Tensor] = None,
        durations: Optional[Tensor] = None,
        mels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = torch.cat(
            (self.embed_tokens(tokens), self.embed_tones(tones)), dim=-1
        ).transpose(1, 2)

        for encoder_layer in self.encoder:
            x = encoder_layer(x)
        encoder_outputs = self.layer_norm(x).transpose(1, 2)

        if self.num_speakers > 1:
            encoder_outputs += self.speaker_embedding(speakers.long()).unsqueeze(1)

        duration_prediction = self.duration_predictor(
            encoder_outputs.transpose(1, 2)
        ).squeeze(1)

        if mels is not None and durations is not None:
            durations = torch.clamp(torch.round(durations), min=0).long()
            mel_time = mels.shape[1]
            assert torch.max(torch.sum(durations, dim=1)).item() == mel_time
        else:
            duration_prediction = torch.exp(duration_prediction) - 1
            durations = torch.clamp(torch.round(duration_prediction), min=0).long()
            mel_time = torch.max(torch.sum(durations, dim=1)).long()

        decoder_inp = self._length_regulator(encoder_outputs, mel_time, durations)
        decoder_inp = self.dropout(decoder_inp).transpose(1, 2)

        pitch_feat = self.pitch_predictor(decoder_inp)
        new_feat = (
            torch.stack((pitches, periodicity), dim=2).transpose(1, 2)
            if pitches is not None
            else pitch_feat.clone()
        )
        new_feat = new_feat.detach()

        decoder_inp += self.embed_pitch(new_feat)

        for decoder_layer in self.decoder:
            decoder_inp = decoder_layer(decoder_inp)
        decoder_outputs = self.layer_norm2(decoder_inp)

        decoder_outputs = self.mel_out(decoder_outputs).transpose(1, 2)

        return decoder_outputs, duration_prediction, pitch_feat[:, 0], pitch_feat[:, 1]


class PitchDatasetSpeechSynth(PitchDataset):
    """
    Implementation of PitchDataset using synthetic speech generation with LightSpeech.

    This dataset generates synthetic speech samples using a LightSpeech TTS model and vocoder,
    providing audio along with corresponding pitch and periodicity labels. The dataset
    generates samples for each speaker, using Chinese words.

    Args:
        root_file (str): Path to the LightSpeech model file. Defaults to "lightspeech.pt"
        samples_per_speaker (int): Number of samples per speaker. Defaults to 10
        word_range (Tuple): Range of words per generated sample. Defaults to (3, 9)
        periodicity_threshold (float): Threshold for periodicity detection. Defaults to 0.4
        device (str): Device to run inference on. Defaults to "cuda"
        vocoder_name (str): Vocoder model name. Defaults to "hifigan_lj_ft_t2_v1"
        use_cache (bool): Whether to cache generated samples. Defaults to True
        **kwargs: Additional arguments passed to PitchDataset
    """

    def __init__(
        self,
        root_dir: str = "lightspeech.pt",
        samples_per_speaker: int = 10,
        word_range: Tuple = (3, 9),
        periodicity_threshold: float = 0.4,
        device: str = "cuda",
        vocoder_name: str = "hifigan_lj_ft_t2_v1",
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_file = Path(root_dir)
        self.samples_per_speaker = samples_per_speaker
        self.word_range = word_range
        self.periodicity_threshold = periodicity_threshold
        self.device = torch.device(device)
        self.vocoder_name = vocoder_name
        self.use_cache = use_cache
        self.data_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        self._initialize_models()

        # Calculate total dataset size
        self.total_samples = self.num_speakers * self.samples_per_speaker

    def _initialize_models(self):
        """Initialize the TTS model and vocoder."""
        try:
            state_dict = torch.load(
                self.model_file, map_location=self.device, weights_only=False
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file '{self.model_file}' not found")
        except Exception as e:
            raise IOError(f"Error loading model file '{self.model_file}': {str(e)}")

        # Initialize LightSpeech model
        self.tts_model = (
            LightSpeech(
                num_phones=state_dict["num_phones"],
                num_speakers=state_dict["num_speakers"],
                num_mel_bins=state_dict["num_mel_bins"],
            )
            .to(self.device)
            .eval()
        )
        self.tts_model.load_state_dict(state_dict["state_dict"], strict=True)

        # Initialize vocoder
        try:
            self.vocoder = torch.hub.load(
                "lars76/bigvgan-mirror",
                self.vocoder_name,
                trust_repo=True,
                pretrained=True,
                verbose=False,
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Error loading vocoder: {str(e)}")

        # Store model parameters and dictionaries
        self.num_speakers = state_dict["num_speakers"]
        self.pinyin_to_ipa = state_dict["pinyin_dict"]
        self.ipa_to_token = state_dict["phone_dict"]
        self.speaker_info = state_dict["speaker_dict"]

        if not self.ipa_to_token:
            raise ValueError("Phone dictionary is empty in the loaded model")

        # Extract available pinyin words (excluding special tokens)
        self.available_words = [
            word
            for word in self.pinyin_to_ipa.keys()
            if not word.startswith("<") and not word.endswith(">")
        ]

        if not self.available_words:
            raise ValueError("No valid words found in pinyin dictionary")

    def _convert_pinyin_to_ipa(self, pinyin_text: str) -> str:
        """Convert Pinyin text to IPA notation with tones."""
        ipa_string = ""

        for syllable in pinyin_text.split():
            syllable = syllable.strip()

            # Ensure each syllable ends with a digit for tone
            if not syllable[-1].isdigit():
                if syllable == "<sil>":
                    syllable += "1"
                else:
                    syllable += "5"  # Default to neutral tone

            ipa_key = syllable[:-1]
            tone = syllable[-1]

            ipas = self.pinyin_to_ipa.get(ipa_key)
            if ipas is None:
                continue

            ipa_string += ipas.replace(" ", "") + tone + " "

        return ipa_string.strip()

    def _convert_ipa_to_tokens(self, ipa_text: str) -> Tuple[List[int], List[int]]:
        """Convert IPA text to token and tone IDs."""
        token_ids = []
        tone_ids = []

        # Sort phonemes by length in descending order for proper matching
        sorted_phonemes = sorted(self.ipa_to_token.keys(), key=len, reverse=True)

        for token in ipa_text.split():
            if token[-1].isdigit():
                tone_id = int(token[-1]) + 1
                ipa_key = token[:-1]
            else:
                continue

            i = 0
            while i < len(ipa_key):
                matched = False
                for phoneme in sorted_phonemes:
                    if ipa_key[i:].startswith(phoneme):
                        token_ids.append(self.ipa_to_token[phoneme])
                        tone_ids.append(tone_id)
                        i += len(phoneme)
                        matched = True
                        break
                if not matched:
                    break

        return token_ids, tone_ids

    def _generate_word_sequence(self) -> Tuple[List[int], List[int]]:
        """Generate a sequence of real words and convert to tokens."""
        # Select random number of words within range
        num_words = random.randint(*self.word_range)
        selected_words = random.sample(
            self.available_words, min(num_words, len(self.available_words))
        )

        # Create pinyin text with default tones
        pinyin_text = " ".join(selected_words)

        # Convert to IPA
        ipa_text = self._convert_pinyin_to_ipa(pinyin_text)

        # Convert to tokens
        token_ids, tone_ids = self._convert_ipa_to_tokens(ipa_text)

        return token_ids, tone_ids

    def get_group(self, idx: int) -> str:
        """Return group identifier for sample (speaker ID)"""
        speaker_id = idx // self.samples_per_speaker
        return f"speaker_{speaker_id}"

    def __len__(self) -> int:
        return self.total_samples

    @torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()
    )
    @torch.inference_mode()
    def _generate_tts_sample(
        self, speaker_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a single TTS sample with audio, pitch, and periodicity."""
        # Generate word sequence
        token_ids, tone_ids = self._generate_word_sequence()

        # Always add silence at beginning and end
        sil_token = self.ipa_to_token["<sil>"]
        token_ids = [sil_token] + token_ids + [sil_token]
        tone_ids = [1] + tone_ids + [1]

        # Convert to tensors
        speaker_id_tensor = torch.tensor([speaker_id], dtype=torch.long).to(self.device)
        tokens_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        tone_ids_tensor = torch.tensor([tone_ids], dtype=torch.long).to(self.device)

        try:
            # Generate mel spectrogram and features
            mel, dur, pitch, periodicity = self.tts_model(
                speaker_id_tensor, tokens_tensor, tone_ids_tensor
            )

            # Generate waveform
            wav = self.vocoder(mel.transpose(1, 2)).flatten().cpu()

            # Convert pitch
            pitch = 75.3236 * pitch.flatten().cpu() + 224.5344
            periodicity = (
                periodicity.flatten().cpu() > self.periodicity_threshold
            ).float()

            # Process the sample using parent class method
            wav, pitch, periodicity = self.process_sample(
                wav, pitch, periodicity, self.vocoder.sampling_rate
            )

            return wav, pitch, periodicity

        except Exception as e:
            raise RuntimeError(f"Error during TTS generation: {str(e)}")

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        if not 0 <= idx < len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        # Calculate which speaker this sample belongs to
        speaker_id = idx // self.samples_per_speaker
        sample_idx_for_speaker = idx % self.samples_per_speaker

        if idx not in self.data_cache or not self.use_cache:
            try:
                waveform, pitch, periodicity = self._generate_tts_sample(speaker_id)

                if self.use_cache:
                    self.data_cache[idx] = (waveform, pitch, periodicity)
            except Exception as e:
                raise RuntimeError(f"Error generating sample at index {idx}: {str(e)}")
        else:
            waveform, pitch, periodicity = self.data_cache[idx]

        return {
            "audio": waveform.float(),
            "pitch": pitch.float(),
            "periodicity": periodicity,
            "speaker_id": speaker_id,
            "sample_idx": idx,
        }
