

import torch
from architecture import GptModel, load_weights_into_gpt


BASE_CONFIG = {
    "vocab_size": 50257,     # GPT-2 BPE vocabulary size
    "context_length": 1024,  # GPT-2's trained max context
    "drop_rate": 0.0,        # no dropout - fine-tuning on small datasets, short runs
    "qkv_bias": True,        # GPT-2's c_attn has bias terms - required for correct weight loading
}

MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def instantiate_model(choose_model, load_weights):

    cfg = dict(BASE_CONFIG)
    cfg.update(MODEL_CONFIGS[choose_model])

    if not load_weights:
        torch.manual_seed(123)
    model = GptModel(cfg) 

    if load_weights:
        from gpt2_downloader import download_and_load_gpt2  

        model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2_weights")
        load_weights_into_gpt(model, params)

    model.eval()
    return model, cfg
