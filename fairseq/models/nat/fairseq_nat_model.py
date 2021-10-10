# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
)

from fairseq.distributed import fsdp_wrap
from fairseq.modules import transformer_layer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper

from fairseq.modules.hcg_attention import MultiHeadHCGAttention
from fairseq.modules.transformer_sentence_encoder import init_bert_params


def ensemble_encoder(func):
    def wrapper(self, *args, **kwargs):
        if self.ensemble_models is None or len(self.ensemble_models) == 1:
            return func(self, *args, **kwargs)
        encoder_outs = [func(model, *args, **kwargs, return_all_hiddens=True) for model in self.ensemble_models]
        _encoder_out = encoder_outs[0].copy()

        def stack(key):
            outs = [e[key][0] for e in encoder_outs]
            return [torch.stack(outs, -1) if outs[0] is not None else None]

        _encoder_out["encoder_out"] = stack("encoder_out")
        _encoder_out["encoder_embedding"] = stack("encoder_embedding")

        num_layers = len(_encoder_out["encoder_states"])
        if num_layers > 0:
            _encoder_out["encoder_states"] = [
                torch.stack([e["encoder_states"][i] for e in encoder_outs], -1)
                for i in range(num_layers)
            ]
        return _encoder_out

    return wrapper


def ensemble_decoder(func):
    def wrapper(self, normalize=False, encoder_out=None, *args, **kwargs):
        if self.ensemble_models is None or len(self.ensemble_models) == 1:
            return func(
                self, normalize=normalize, encoder_out=encoder_out, *args, **kwargs
            )

        def _replace(encoder_out, new_val):
            new_encoder_out = encoder_out.copy()
            new_encoder_out["encoder_out"] = [new_val]
            return new_encoder_out

        action_outs = [
            func(
                model,
                normalize=normalize,
                encoder_out=_replace(
                    encoder_out,
                    encoder_out["encoder_out"][0][:, :, :, i]
                ),
                *args,
                **kwargs
            )
            for i, model in enumerate(self.ensemble_models)
        ]

        if not isinstance(action_outs[0], tuple):  # return multiple values
            action_outs = [[a] for a in action_outs]
        else:
            action_outs = [list(a) for a in action_outs]

        ensembled_outs = []
        for i in range(len(action_outs[0])):
            if i == 0 and normalize:
                ensembled_outs += [
                    torch.logsumexp(
                        torch.stack([a[i] for a in action_outs], -1), dim=-1
                    )
                    - math.log(len(self.ensemble_models))
                ]
            elif action_outs[0][i] is not None:
                ensembled_outs += [torch.stack([a[i] for a in action_outs], -1)]
            else:
                ensembled_outs += [None]

        if len(ensembled_outs) == 1:
            return ensembled_outs[0]
        return tuple(ensembled_outs)

    return wrapper


class FairseqNATModel(TransformerModel):
    """
    Abstract class for all nonautoregressive-based models
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.tgt_dict = decoder.dictionary
        self.bos = decoder.dictionary.bos()
        self.eos = decoder.dictionary.eos()
        self.pad = decoder.dictionary.pad()
        self.unk = decoder.dictionary.unk()

        self.ensemble_models = None

    @property
    def allow_length_beam(self):
        return False

    @property
    def allow_ensemble(self):
        return True

    def enable_ensemble(self, models):
        self.encoder.ensemble_models = [m.encoder for m in models]
        self.decoder.ensemble_models = [m.decoder for m in models]

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--apply-bert-init",
            action="store_true",
            help="use custom param initialization for BERT",
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = FairseqNATDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = FairseqNATEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    def forward_encoder(self, encoder_inputs):
        return self.encoder(*encoder_inputs)

    def forward_decoder(self, *args, **kwargs):
        return NotImplementedError

    def initialize_output_tokens(self, *args, **kwargs):
        return NotImplementedError

    def forward(self, *args, **kwargs):
        return NotImplementedError



class TransformerEncoderLayerHCG(transformer_layer.TransformerEncoderLayerBase):
    def __init__(self, args):
        self.with_hard_concrete_gate = getattr(args, "with_hard_concrete_gate", False)
        super().__init__(args)

        return

    def build_self_attention(self, embed_dim, cfg):
        # print("TransformerEncoderLayerHCG: build_self_attention")
        return MultiHeadHCGAttention(embed_dim, cfg.encoder.attention_heads, with_hard_concrete_gate=self.with_hard_concrete_gate)


class FairseqNATEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.ensemble_models = None

    @ensemble_encoder
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

class FairseqNATEncoderHCG(FairseqNATEncoder):

    def build_encoder_layer(self, cfg):
        layer = TransformerEncoderLayerHCG(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


class TransformerDecoderLayerHCG(transformer_layer.TransformerDecoderLayerBase):
    def build_self_attention(self, embed_dim, cfg, **kwarg):
        # print("TransformerDecoderLayerHCG: build_self_attention")
        return MultiHeadHCGAttention(embed_dim, cfg.decoder.attention_heads, with_hard_concrete_gate=kwarg.get("with_hard_concrete_gate", False))

    def build_encoder_attention(self, embed_dim, cfg, **kwarg):
        return MultiHeadHCGAttention(embed_dim, cfg.decoder.attention_heads, with_hard_concrete_gate=kwarg.get("with_hard_concrete_gate", False))
        # return super().build_encoder_attention(
        #     embed_dim,
        #     TransformerConfig.from_namespace(args),
        # )

class FairseqNATDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.ensemble_models = None

    def build_self_attention(self, embed_dim, cfg):
        return MultiHeadHCGAttention(embed_dim, cfg.decoder.attention_heads, with_hard_concrete_gate=self.args.with_hard_concrete_gate)

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = transformer_layer.TransformerDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer