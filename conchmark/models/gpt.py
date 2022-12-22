import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel


class GPTLMModel(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        max_seq_len=1024,
        vocab_size=2048,
        checkpoint=False,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(
            GPT2Config(
                n_embd=hidden_size,
                n_layer=num_layers,
                n_head=num_attention_heads,
                n_positions=max_seq_len,
                n_ctx=max_seq_len,
                vocab_size=vocab_size,
            )
        )
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=not self.checkpoint,
        )[0]


class GPTLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )


def gpt2_medium(checkpoint=False):
    return GPTLMModel(
        hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=checkpoint
    )


def gpt2_xl(checkpoint=True):
    return GPTLMModel(
        hidden_size=1600, num_layers=48, num_attention_heads=32, checkpoint=checkpoint
    )


def gpt2_4b(checkpoint=True):
    return GPTLMModel(
        hidden_size=2304, num_layers=64, num_attention_heads=16, checkpoint=checkpoint
    )


def gpt2_6b(checkpoint=True):
    return GPTLMModel(
        hidden_size=4096, num_layers=30, num_attention_heads=16, checkpoint=checkpoint
    )


def gpt2_8b(checkpoint=True):
    return GPTLMModel(
        hidden_size=3072, num_layers=72, num_attention_heads=24, checkpoint=checkpoint
    )


def gpt2_10b(checkpoint=True):
    return GPTLMModel(
        hidden_size=4096, num_layers=50, num_attention_heads=16, checkpoint=checkpoint
    )


def gpt2_12b(checkpoint=True):
    return GPTLMModel(
        hidden_size=4096, num_layers=60, num_attention_heads=16, checkpoint=checkpoint
    )


def gpt2_14b(checkpoint=True):
    return GPTLMModel(
        hidden_size=4096, num_layers=70, num_attention_heads=16, checkpoint=checkpoint
    )


def gpt2_28b(checkpoint=True):
    return GPTLMModel(
        hidden_size=8192, num_layers=35, num_attention_heads=16, checkpoint=checkpoint
    )


def gpt2_32b(checkpoint=True):
    return GPTLMModel(
        hidden_size=8192, num_layers=40, num_attention_heads=16, checkpoint=checkpoint
    )


def gpt2_36b(checkpoint=True):
    return GPTLMModel(
        hidden_size=8192, num_layers=45, num_attention_heads=16, checkpoint=checkpoint
    )
