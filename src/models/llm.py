import torch
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

    @torch.compiler.disable
    def loop_layers(self, embeddings, attention_mask, num_steps_pair):
        num_steps_no_grad, num_steps_with_grad = map(int, num_steps_pair)
        outputs = embeddings

        with torch.no_grad():
            for _ in range(num_steps_no_grad):
                outputs = self.model(outputs, mask = attention_mask, causal_mask = True)

        if num_steps_no_grad:
            # Keep embeddings in DDP's graph; TBPTT intentionally makes this gradient zero.
            outputs = outputs + embeddings.sum() * 0

        for _ in range(num_steps_with_grad):
            outputs = self.model(outputs, mask = attention_mask, causal_mask = True)

        return outputs

    def forward(self, batch, **kwargs):
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)

        if attention_mask is not None:
            attention_mask = prepare_attention_mask(attention_mask)

        embeddings = self.token_embeddings(input_ids)

        num_loops = batch.get('num_loops', self.args.model_args.get('mean_recurrence', 1))
        num_steps_pair = batch.get('num_steps_pair', (0, num_loops))
        outputs = self.loop_layers(embeddings, attention_mask, num_steps_pair)

        out = self.decoder_out(outputs)

        return out
