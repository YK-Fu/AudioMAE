from functools import partial

import torch
import torch.nn as nn

from omegaconf.dictconfig import DictConfig
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_lm_encoder_decoder_model import MegatronLMEncoderDecoderModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group
)
from nemo.utils import logging

from dataset import AudioDataset
from modules import PretrainAudioMAEModule
        
class PretrainAudioMAEModel(MegatronLMEncoderDecoderModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.patch_size = self._cfg.patch_size
    @property
    def model_name(self):
        return "AudioMAE"

    # TODO
    def model_provider_func(self, pre_process, post_process, **kwargs):
        model = PretrainAudioMAEModule(
            config=self.model_parallel_config,
            encoder_cfg=self.cfg.encoder,
            decoder_cfg=self.cfg.decoder,
            precision=self.trainer.precision,
            mask_prob=self.cfg.mask_prob,
            seq_length=self.cfg.seq_length
        )

        return model

    # TODO
    def build_train_valid_test_datasets(self):
        logging.info(f'Building {self.model_name} datasets.')
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")
        global_batch_size = self._cfg.global_batch_size
        eval_iters = (self.trainer.max_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches
        train_valid_test_num_samples = [
            self.trainer.max_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]
        self._train_ds = AudioDataset(
            dataset_json_file=self._cfg.data.data_path + '/' + self._cfg.data.data_prefix['train'] + '.jsonl',
            patch_size=self._cfg.patch_size,
            mask_prob=self._cfg.mask_prob,
            mel_bins=self._cfg.data.mel_bins,
            target_length=self._cfg.data.target_length
        )
        self._validation_ds = AudioDataset(
            dataset_json_file=self._cfg.data.data_path + '/' + self._cfg.data.data_prefix['valid'] + '.jsonl',
            patch_size=self._cfg.patch_size,
            mask_prob=self._cfg.mask_prob,
            mel_bins=self._cfg.data.mel_bins,
            target_length=self._cfg.data.target_length
        )
        self._test_ds = AudioDataset(
            dataset_json_file=self._cfg.data.data_path + '/' + self._cfg.data.data_prefix['test'] + '.jsonl',
            patch_size=self._cfg.patch_size,
            mask_prob=self._cfg.mask_prob,
            mel_bins=self._cfg.data.mel_bins,
            target_length=self._cfg.data.target_length
        )
        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building {self.model_name} datasets.')
        self._reconfigure_limit_batches(self.trainer.limit_val_batches, self._validation_dl, 'val')

        return self._train_ds, self._validation_ds, self._test_ds
    # TODO
    def get_forward_output_only_fuc(self, arg_names, output_name, **kwargs):
        def fwd_output_only_func(dataloader_iter, model):
            raise NotImplemented
        return fwd_output_only_func

    def get_labels(self, input_ids):
        p = self.patch_size
        b = input_ids.size(0)
        h = input_ids.size(2) // p
        w = input_ids.size(3) // p
        x = input_ids.reshape(shape=(b, 1, h, p, w, p))
        x = torch.einsum('bchpwq->bhwpqc', x)
        x = x.reshape(shape=(b, h*w, p*p*1))
        return x

    # TODO
    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(dataloader_iter, model):
            (input_ids, mask, enc_attn_mask, dec_attn_mask, position_ids), _, _ = next(dataloader_iter)
            input_ids = input_ids.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            enc_attn_mask = enc_attn_mask.cuda(non_blocking=True)
            dec_attn_mask = dec_attn_mask.cuda(non_blocking=True)
            position_ids = position_ids.cuda(non_blocking=True)
            labels = self.get_labels(input_ids)
            output_tensor = model(
                enc_input_ids=input_ids,
                mask=mask,
                dec_input_ids=input_ids,
                enc_attn_mask=enc_attn_mask,
                dec_attn_mask=dec_attn_mask,
                position_ids=position_ids,
                labels=labels,
            ).contiguous()

            def loss_func(output_tensor):
                loss_mask = ~mask
                loss = torch.sum(output_tensor[loss_mask].view(-1).float()) / loss_mask.sum()
                reduced_loss = average_losses_across_data_parallel_group([loss])
                loss_dict = {'loss': reduced_loss}
                return loss, loss_dict

            return output_tensor, loss_func

        return fwd_output_and_loss_func

