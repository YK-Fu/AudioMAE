import json
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(
        self,
        dataset_json_file: str,
        patch_size: int,
        mask_prob: float,
        mel_bins: int,
        target_length: int,
        norm_mean=-4.2677393,
        norm_std=4.5689974
    ):
        self.samples = []
        with open(dataset_json_file) as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    data = json.loads(line)
                    self.samples.append([data['audio_filepath'], data['duration']])
        self.patch_size = patch_size
        self.mask_prob = mask_prob
        self.mel_bins = mel_bins
        self.target_length = target_length // (mel_bins // patch_size)
        self.total = len(self.samples)
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self):
        return self.total

    def _wav2fbank(self, filename):
        waveform, sr = torchaudio.load('/bcp/workspaces/hackathon/' + filename)
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.mel_bins, dither=0.0, frame_shift=10)
        n_frames = fbank.shape[0]
        if self.target_length - n_frames > 0:
            fbank = torch.nn.ZeroPad2d((0, 0, 0, self.target_length - n_frames))(fbank)
        else:
            fbank = fbank[:self.target_length, :]
        return fbank

    def __getitem__(self, index: int):
        path, dur = self.samples[index]
        fbank = self._wav2fbank(path)
        fbank = fbank.transpose(0, 1).unsqueeze(0)
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        L = (self.mel_bins // self.patch_size) * (self.target_length // self.patch_size)
        len_keep = int(L * (1 - self.mask_prob))
        noise = torch.randn(L)

        ids_shuffle = torch.argsort(noise, dim=0)
        ids_restore = torch.argsort(ids_shuffle, dim=0)
        ids_keep = ids_shuffle[:len_keep]

        mask = torch.ones(L)
        mask[:len_keep] = 0
        mask = torch.gather(mask, dim=0, index=ids_restore) < 0.5

        enc_attn_mask = torch.ones(len_keep).long()

        dec_attn_mask = torch.ones(L).long()
        position_id = torch.arange(0, L).long()

        return fbank, mask, enc_attn_mask, dec_attn_mask, position_id

if __name__ == '__main__':
    dataset = AudioDataset('/bcp/workspaces/hackathon/data/train.jsonl', 16, 0.8, 16, 512)
    dataset[0]