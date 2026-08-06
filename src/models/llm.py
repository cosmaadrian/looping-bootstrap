import torch.nn as nn
from .utils import MuReadout
from .building_blocks import TransformerEncoder


def prepare_attention_mask(attention_mask):
    attention_mask = attention_mask.float()
    attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
    attention_mask[attention_mask == 0] = -10000
    attention_mask[attention_mask == 1] = 0
    return attention_mask


class TransformerDecoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        from lib import nomenclature
        from lib.accelerator import AcumenAccelerator

        self.nomenclature = nomenclature
        self.accelerator = AcumenAccelerator()

        self.token_vocab_size = args.input_tokenizer.vocab_size
        self.token_vocab_size += 256  # for start and end answer tokens, empty token, start of text token, and pad token

        self.token_embeddings = nn.Embedding(
            num_embeddings = self.token_vocab_size,
            embedding_dim = int(self.args.model_args.dmodel * self.args.model_width_multiplier),
        )
        ###################################################################

        self.model = TransformerEncoder(
            args = self.args,
            dmodel = int(self.args.model_args.dmodel * self.args.model_width_multiplier),
            depth = self.args.model_args.num_layers,
            nheads = int(8 * self.args.model_width_multiplier),
            dropout = self.args.model_args.dropout,
            attn_dropout = self.args.model_args.attn_dropout,
            has_context = False,
        )

        self.decoder_out = MuReadout(
            in_features = int(self.args.model_args.dmodel * self.args.model_width_multiplier),
            out_features = self.token_vocab_size,
            args = args,
        )

    def forward(self, batch, **kwargs):
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)

        if attention_mask is not None:
            attention_mask = prepare_attention_mask(attention_mask)

        embeddings = self.token_embeddings(input_ids)

        outputs = self.model(
            embeddings,
            mask = attention_mask,
            causal_mask = True,
        )

        out = self.decoder_out(outputs)

        return out