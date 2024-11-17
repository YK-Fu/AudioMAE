import math
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from omegaconf.dictconfig import DictConfig

from typing import Any, Dict, Optional, Union
from nemo.collections.nlp.modules.common.megatron.megatron_encoder_decoder import (
    MegatronTransformerEncoderDecoderModule,
)
from nemo.collections.nlp.modules.common.megatron.megatron_encoder_module import MegatronEncoderModule
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.megatron_decoders import get_decoder_model
from nemo.collections.nlp.modules.common.megatron.megatron_encoders import get_encoder_model
from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType
from nemo.collections.vision.modules.common.megatron.vision_transformer import ParallelVisionTransformer
from nemo.core.classes.exportable import Exportable
from nemo.utils import logging


from megatron.core import ModelParallelConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal

from apex.transformer.enums import AttnMaskType, ModelType

def get_positional_embedding(hidden_size, length):
    if hidden_size % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(hidden_size))
    pe = torch.zeros(length, hidden_size)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, hidden_size, 2, dtype=torch.float) *
                         -(math.log(10000.0) / hidden_size)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    positional_embedding = nn.Embedding.from_pretrained(pe, freeze=True)
     
    return positional_embedding

class PretrainAudioMAEModule(MegatronModule):
    def __init__(
        self,
        config: ModelParallelConfig,
        encoder_cfg: DictConfig,
        decoder_cfg: DictConfig,
        seq_length=512,
        mask_prob=0.8,
        pre_process=True,
        post_process=True,
        precision=16,
        share_input_params=True,
        decoder_head_bias=True,
    ):
        super().__init__(config=config)
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.pre_process = pre_process
        self.post_process = post_process
        self.precision = precision
        self.decoder_head_bias = decoder_head_bias
        self.share_input_params = share_input_params
        self.mask_prob = mask_prob

        # used for decoder input
        self.mask_embedding = nn.Parameter(torch.randn(1, decoder_cfg.hidden_size))
        self.position_embedding = get_positional_embedding(encoder_cfg.hidden_size, seq_length)

        encoder_kv_channels, decoder_kv_channels = self._validate_config()
        if pre_process:
            # TODO parallel linear project from patch dim to transformer input
            self.encoder_proj = nn.Conv2d(1, encoder_cfg.hidden_size, kernel_size=encoder_cfg.patch_size, stride=encoder_cfg.patch_size)
            if hasattr(self, 'encoder_proj') and self.share_input_params:
                assert encoder_cfg.patch_size == decoder_cfg.patch_size, "patch size should be the same for share projection layer"
                self.decoder_proj = self.encoder_proj
            else:
                self.decoder_proj = nn.Conv2d(1, decoder_cfg.hidden_size, kernel_size=decoder_cfg.patch_size, stride=decoder_cfg.patch_size)
                if self.share_input_params:
                    self.decoder_proj.zero_parameters()
        # should we always wrap encoder into MegatronEncoderModule?
        encoder = get_encoder_model(
            config=config,
            arch=encoder_cfg.arch,
            init_method=init_method_normal(encoder_cfg.get('init_method_std', 0.02)),
            scaled_init_method=scaled_init_method_normal(
                encoder_cfg.get('init_method_std', 0.02), encoder_cfg.num_layers
            ),
            init_method_std=encoder_cfg.get('init_method_std', 0.02),
            num_layers=encoder_cfg.num_layers,
            hidden_size=encoder_cfg.hidden_size,
            num_attention_heads=encoder_cfg.num_attention_heads,
            apply_query_key_layer_scaling=encoder_cfg.apply_query_key_layer_scaling,
            kv_channels=encoder_cfg.kv_channels,
            ffn_hidden_size=encoder_cfg.ffn_hidden_size,
            # self_attn_mask_type=self.encoder_attn_mask_type, # TODO (yuya)
            pre_process=self.pre_process,
            post_process=self.post_process,
            precision=encoder_cfg.precision,
            fp32_residual_connection=encoder_cfg.fp32_residual_connection,
            activations_checkpoint_method=encoder_cfg.activations_checkpoint_method,
            activations_checkpoint_num_layers=encoder_cfg.activations_checkpoint_num_layers,
            normalization=encoder_cfg.normalization,
            layernorm_epsilon=encoder_cfg.layernorm_epsilon,
            hidden_dropout=encoder_cfg.hidden_dropout,
            attention_dropout=encoder_cfg.attention_dropout,
            bias_activation_fusion=encoder_cfg.get("bias_activation_fusion", False),
            persist_layer_norm=encoder_cfg.persist_layer_norm,
            openai_gelu=encoder_cfg.openai_gelu,
            onnx_safe=encoder_cfg.onnx_safe,
            masked_softmax_fusion=encoder_cfg.masked_softmax_fusion,
            megatron_legacy=encoder_cfg.megatron_legacy,
            activations_checkpoint_granularity=encoder_cfg.activations_checkpoint_granularity,
            activation=encoder_cfg.get('activation', 'gelu'),
            use_flash_attention=encoder_cfg.get('use_flash_attention', False),
        )

        decoder = get_decoder_model(
            config=config,
            arch=decoder_cfg.arch,
            hidden_size=decoder_cfg.hidden_size,
            ffn_hidden_size=decoder_cfg.ffn_hidden_size,
            num_layers=decoder_cfg.num_layers,
            num_attention_heads=decoder_cfg.num_attention_heads,
            apply_query_key_layer_scaling=decoder_cfg.get('apply_query_key_layer_scaling', True),
            kv_channels=decoder_kv_channels,
            init_method=init_method_normal(decoder_cfg.get('init_method_std', 0.02)),
            scaled_init_method=scaled_init_method_normal(
                decoder_cfg.get('init_method_std', 0.02), decoder_cfg.num_layers
            ),
            decoder_attn_mask_type=AttnMaskType.causal,
            pre_process=pre_process,
            post_process=post_process,
            init_method_std=decoder_cfg.get('init_method_std', 0.02),
            hidden_dropout=decoder_cfg.get('hidden_dropout', 0.1),
            attention_dropout=decoder_cfg.get('attention_dropout', 0.1),
            ffn_dropout=decoder_cfg.get('ffn_dropout', 0.0),
            precision=precision,
            fp32_residual_connection=decoder_cfg.get('fp32_residual_connection', False),
            activations_checkpoint_method=decoder_cfg.get('activations_checkpoint_method', None),
            activations_checkpoint_num_layers=decoder_cfg.get('activations_checkpoint_num_layers', 1),
            activations_checkpoint_granularity=decoder_cfg.get('activations_checkpoint_granularity', None),
            layernorm_epsilon=decoder_cfg.get('layernorm_epsilon', 1e-5),
            bias_activation_fusion=decoder_cfg.get('bias_activation_fusion', True),
            bias_dropout_add_fusion=decoder_cfg.get('bias_dropout_add_fusion', True),
            masked_softmax_fusion=decoder_cfg.get('masked_softmax_fusion', True),
            persist_layer_norm=decoder_cfg.get('persist_layer_norm', True),
            openai_gelu=decoder_cfg.get('openai_gelu', False),
            onnx_safe=decoder_cfg.get('onnx_safe', False),
            hidden_steps=decoder_cfg.get('hidden_steps', -1),
            activation=decoder_cfg.get('activation', 'gelu'),
            bias=decoder_cfg.get('bias', True),
            normalization=decoder_cfg.get('normalization', 'layernorm'),
            transformer_block_type=decoder_cfg.get('transformer_block_type', 'pre_ln'),
            headscale=decoder_cfg.get('headscale', False),
            parent_model_type=ModelType.encoder_and_decoder,
            megatron_legacy=decoder_cfg.get('megatron_legacy', False),
            normalize_attention_scores=decoder_cfg.get('normalize_attention_scores', True),
            num_moe_experts=decoder_cfg.get('num_moe_experts', 1),
            moe_frequency=decoder_cfg.get('moe_frequency', 1),
            moe_dropout=decoder_cfg.get('moe_dropout', 0.0),
            position_embedding_type=decoder_cfg.get('position_embedding_type', 'learned_absolute'),
            use_flash_attention=decoder_cfg.get('use_flash_attention', False),
        )
        self.enc_dec_model = MegatronTransformerEncoderDecoderModule(
            config=config,
            encoder=encoder,
            decoder=decoder,
        )
        if self.share_input_params:
            # TODO, PP > 1
            pass
        if post_process:
            self.decoder_head = nn.Linear(decoder_cfg.hidden_size, decoder_cfg.patch_size ** 2, bias=True)

            
    def _validate_kv_channels(self, cfg):
        kv_channels = cfg.kv_channels
        if cfg.kv_channels is None:
            assert (
                cfg.hidden_size % cfg.num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = cfg.hidden_size // cfg.num_attention_heads

        return kv_channels

    def _validate_config(self):
        encoder_kv_channels = self._validate_kv_channels(self.encoder_cfg)
        decoder_kv_channels = self._validate_kv_channels(self.decoder_cfg)
        self._validate_enc_dec_hidden_size(self.encoder_cfg, self.decoder_cfg)
        # if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        #     assert (
        #         self.share_token_embeddings
        #     ), "Token embeddings must be shared when using pipeline model parallel size > 1"
        #     assert (
        #         self.share_decoder_tokens_head_embeddings
        #     ), "Decoder token embeddings and the outputlayer must be shared when using pipeline model parallel size > 1"


        return encoder_kv_channels, decoder_kv_channels

    def _validate_enc_dec_hidden_size(self, encoder_cfg, decoder_cfg):
        if encoder_cfg.hidden_size != decoder_cfg.hidden_size:
            raise ValueError(
                f"Encoder and decoder hidden_size must be equal, but got encoder: {encoder_cfg.hidden_size} and decoder: {decoder_cfg.hidden_size}"
            )
    def _patchify(self, x):
        pass

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None

        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert (
            len(input_tensor) == 1
        ), 'input_tensor should only be length 1 for stage with both encoder and decoder'
        self.enc_dec_model.encoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        enc_input_ids=None,
        enc_attn_mask=None,
        dec_input_ids=None,
        dec_attn_mask=None,
        position_ids=None,
        labels=None,
        enc_output=None,
        enc_output_attn_mask=None,
        enc_input=None,
        mask=None,
        output_enc_hidden_only=False,
    ):
        if enc_input is not None and enc_output is not None:
            raise ValueError(
                """Both enc_input and enc_output are not None.
                You should only be passing one of them.
                enc_input is the result of the encoder projection layer
                enc_output is the result of running the entire transformer encoder."""
            )

        # In order of precedence, we use enc_output, enc_input, and then enc_input_ids to determine the encoder sequence length.
        if enc_output is not None:
            # If enc_output is provided in `batch_for_pipeline`, we need to transpose it from [B x S x H] -> [S x B x H].
            enc_output = enc_output.transpose(0, 1)
            enc_seq_length = enc_output.size(0)
        elif enc_input is not None:
            # If enc_input is provided, we need to transpose it from [B x S x H] -> [S x B x H].
            enc_input = enc_input.transpose(0, 1)
            enc_seq_length = enc_input.size(0)
        # Only need to run encoder embedding and position ids if enc_input or enc_output is not provided.
        elif enc_input_ids is not None:
            if self.pre_process:
                assert mask is not None, "mask is none"
                enc_input = self.encoder_proj(enc_input_ids).float()
                b, d, h, w = enc_input.size()
                enc_input = enc_input.view(b, d, -1).permute(0, 2, 1)
                enc_input += self.position_embedding(position_ids)
                enc_input = enc_input[mask].view(b, -1, d).transpose(0, 1)
            else:
                enc_input = None
        else:
            raise ValueError(
                """Enc_input, enc_output, enc_input_ids are all None"""
            )

        if output_enc_hidden_only:
            # When pipeline parallel > 1 we need to make sure encoder exist (will be missing in decoder)
            if enc_output is None and self.enc_dec_model.encoder is not None:
                enc_output = self.enc_dec_model.encode(
                    enc_input=enc_input,
                    enc_attn_mask=enc_attn_mask,
                    enc_layer_past=None,
                    enc_get_key_value=False,
                    enc_self_attention_relative_position_bias=encoder_self_attention_relative_position_bias,
                    batch_data=batch_data,
                )
            else:
                enc_output = self.enc_dec_model.encoder_hidden_state

            return enc_output
        else:
            if enc_output_attn_mask is None:
                enc_output_attn_mask = enc_attn_mask

            if self.pre_process:
                assert mask is not None
                dec_input = self.decoder_proj(dec_input_ids).float()
                b, d, h, w = dec_input.size()
                dec_input = dec_input.view(b, d, -1).permute(0, 2, 1)
                dec_input[~mask] = self.mask_embedding
                dec_input += self.position_embedding(position_ids)
                dec_input = dec_input.transpose(0, 1)
            else:
                # Note: This is when the decoder itself is split across PP ranks.
                dec_input = None

            output = self.enc_dec_model(
                enc_input=enc_input,
                enc_attn_mask=enc_attn_mask,
                dec_input=dec_input,
                dec_attn_mask=dec_attn_mask,
                enc_layer_past=None,
                enc_get_key_value=False,
                enc_output=enc_output,
                enc_output_attn_mask=enc_output_attn_mask,
                dec_layer_past=None,
                dec_get_key_value=False,
            )

            if self.post_process:
                # TODO
                dec_output, enc_output = output  # [s, b, h]
                token_logits = self.decoder_head(dec_output)
                # [s, b, h] -> [b, s, h]
                token_logits = token_logits.transpose(0, 1).contiguous()
                if labels is not None:
                    token_logits = mse_loss(token_logits, labels, reduction='none').mean(-1)
                return token_logits
            else:
                decoder_output, _ = output
                return decoder_output
