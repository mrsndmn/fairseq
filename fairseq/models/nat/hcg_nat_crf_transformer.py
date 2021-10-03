# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import NACRFTransformerModel, base_architecture
from fairseq.modules import DynamicCRF


@register_model("hcg_nacrf_transformer")
class HCGNACRFTransformerModel(NACRFTransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        NACRFTransformerModel.add_args(parser)
        parser.add_argument(
            "--hcg_log_a",
            type=float,
            help="hard concrete hase log a.",
        )

        parser.add_argument(
            "--hcg_temperature",
            type=float,
            help="hard concrete hase log a.",
        )

