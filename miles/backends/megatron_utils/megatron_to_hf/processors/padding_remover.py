import torch

from miles.backends.megatron_utils.misc_utils import strip_param_name_prefix


def remove_padding(name: str, param: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Adjust vocab size for embedding/output layers to match target vocab_size.
    - If param is larger than vocab_size: truncate
    - If param is smaller than vocab_size: pad with zeros
    - Otherwise: return unchanged
    """
    if strip_param_name_prefix(name) in {"embedding.word_embeddings.weight", "output_layer.weight"}:
        current_size = param.shape[0]
        if current_size == vocab_size:
            return param
        elif current_size > vocab_size:
            # Truncate
            return param[:vocab_size]
        else:
            # Pad with zeros
            padding = torch.zeros(vocab_size - current_size, param.shape[1], dtype=param.dtype, device=param.device)
            return torch.cat([param, padding], dim=0)
    return param
