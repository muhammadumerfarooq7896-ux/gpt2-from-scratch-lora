"""
LoRA (Low-Rank Adaptation), implemented from scratch.

Instead of fine-tuning a weight matrix W directly, LoRA freezes W and learns a
low-rank update  ΔW = A @ B  (scaled by alpha/rank), where A and B are much
smaller than W. This lets you adapt a large pretrained model to a new task
while training only a small fraction of its parameters.


"""

import math
import torch


class LoRALayer(torch.nn.Module):
    """
    The low-rank update itself: x -> (alpha / rank) * (x @ A @ B)

    A: (in_dim, rank)   - randomly initialized, like a normal weight
    B: (rank, out_dim)  - zero-initialized, so ΔW = A @ B = 0 at the very start

    The zero-init on B is what makes LoRA a true no-op before any training:
    a freshly LoRA-wrapped model produces IDENTICAL output to the un-wrapped
    base model, since the low-rank update contributes nothing until B moves
    away from zero during training.
    """

    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # similar to standard weight init
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.rank = rank

    def forward(self, x):

        return (self.alpha / self.rank) * (x @ self.A @ self.B)


class LinearWithLoRA(torch.nn.Module):


    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model, rank, alpha):
    """
    Recursively walks the model and replaces every nn.Linear submodule with a
    LinearWithLoRA wrapper in-place.


    """
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)


def freeze_base_model(model):

    for param in model.parameters():
        param.requires_grad = False


def apply_lora(model, rank, alpha):

    freeze_base_model(model)
    replace_linear_with_lora(model, rank=rank, alpha=alpha)


def count_trainable_params(model):

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def verify_lora_is_noop(model, sample_input, rank, alpha):

    model.eval()
    with torch.no_grad():
        out_before = model(sample_input)

    apply_lora(model, rank=rank, alpha=alpha)

    model.eval()
    with torch.no_grad():
        out_after = model(sample_input)

    max_diff = (out_before - out_after).abs().max().item()
    assert torch.allclose(out_before, out_after, atol=1e-6), (
        f"LoRA wrapping changed model output before training (max diff={max_diff}). "
        "This should never happen if B is correctly zero-initialized — check LoRALayer."
    )
    return max_diff
