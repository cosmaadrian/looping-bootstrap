import torch
import torch.nn as nn
from .utils import MuReadout, configure_depth_mup_parameters
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

        self.pre_transformer = TransformerEncoder(
            args = self.args,
            dmodel = int(self.args.model_args.dmodel * self.args.model_width_multiplier),
            depth = 1,
            nheads = int(8 * self.args.model_width_multiplier),
            dropout = self.args.model_args.dropout,
            attn_dropout = self.args.model_args.attn_dropout,
            has_context = False,
        )

        self.post_transformer = TransformerEncoder(
            args = self.args,
            dmodel = int(self.args.model_args.dmodel * self.args.model_width_multiplier),
            depth = 1,
            nheads = int(8 * self.args.model_width_multiplier),
            dropout = self.args.model_args.dropout,
            attn_dropout = self.args.model_args.attn_dropout,
            has_context = False,
        )

        self.model = TransformerEncoder(
            args = self.args,
            dmodel = int(self.args.model_args.dmodel * self.args.model_width_multiplier),
            depth = self.args.model_args.num_layers,
            nheads = int(8 * self.args.model_width_multiplier),
            dropout = self.args.model_args.dropout,
            attn_dropout = self.args.model_args.attn_dropout,
            has_context = False,
            depth_scaled = True,
        )

        self.decoder_out = MuReadout(
            in_features = int(self.args.model_args.dmodel * self.args.model_width_multiplier),
            out_features = self.token_vocab_size,
            args = args,
        )

        # Only the recurrent stack grows with the modeled depth. The embedding,
        # pre/post blocks, and readout retain their base optimizer scaling.
        configure_depth_mup_parameters(self, self.model, self.args)

    @torch.compiler.disable
    def loop_layers(self, embeddings, attention_mask, num_steps_pair, intended_num_loops = None):
        num_steps_no_grad, num_steps_with_grad = map(int, num_steps_pair)
        if intended_num_loops is None:
            intended_num_loops = num_steps_no_grad + num_steps_with_grad

        intended_num_loops = int(intended_num_loops)

        if intended_num_loops < 1:
            raise ValueError('intended_num_loops must be at least 1')

        outputs = embeddings

        initial_state = torch.randn_like(embeddings) * 0.4
        outputs = outputs + initial_state

        with torch.no_grad():
            for _ in range(num_steps_no_grad):
                outputs = self.model(
                    outputs,
                    mask = attention_mask,
                    causal_mask = True,
                    intended_num_loops = intended_num_loops,
                )

        if num_steps_no_grad:
            # Keep embeddings in DDP's graph; TBPTT intentionally makes this gradient zero.
            outputs = outputs + embeddings.sum() * 0

        for _ in range(num_steps_with_grad):
            outputs = self.model(
                outputs,
                mask = attention_mask,
                causal_mask = True,
                intended_num_loops = intended_num_loops,
            )

        if torch.is_grad_enabled() and not num_steps_with_grad:
            # A fully truncated recurrent pass intentionally has no autograd
            # path through self.model. Give DDP zero gradients for those
            # parameters so its reducer can still finish the iteration.
            recurrent_parameter_anchor = sum(
                (parameter.sum() * 0 for parameter in self.model.parameters() if parameter.requires_grad),
                outputs.new_zeros(()),
            )
            outputs = outputs + recurrent_parameter_anchor

        return outputs

    def forward(self, batch, intended_num_loops = None, is_teacher = False, **kwargs):
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)

        if attention_mask is not None:
            attention_mask = prepare_attention_mask(attention_mask)

        embeddings = self.token_embeddings(input_ids)

        if is_teacher and self.args.model_args.noise_std > 0:
            noise = torch.randn_like(embeddings) * self.args.model_args.noise_std
            embeddings = embeddings + noise

        num_loops = batch.get('num_loops', self.args.model_args.get('mean_recurrence', 1))
        num_steps_pair = batch.get('num_steps_pair', (0, num_loops))

        if intended_num_loops is None and 'num_loops' in batch:
            intended_num_loops = num_loops

        outputs = self.pre_transformer(embeddings, mask = attention_mask, causal_mask = True)
        outputs = self.loop_layers(
            outputs,
            attention_mask,
            num_steps_pair,
            intended_num_loops = intended_num_loops,
        )
        outputs = self.post_transformer(outputs, mask = attention_mask, causal_mask = True)

        out = self.decoder_out(outputs)

        return out
