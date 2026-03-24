import logging
import os

from megatron.training.arguments import parse_args, validate_args
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding

__all__ = ["validate_args", "parse_args", "set_default_megatron_args"]

logger = logging.getLogger(__name__)


def set_default_megatron_args(args):
    # always use zero optimizer
    args.use_distributed_optimizer = True
    # TODO: maybe change this after megatron has good fp8 support
    args.bf16 = not args.fp16
    # placeholders
    if args.seq_length is None:
        args.seq_length = 4096

    explicit_max_pos = getattr(args, "max_position_embeddings", None)
    if not explicit_max_pos or explicit_max_pos <= 0:
        args.max_position_embeddings = args.seq_length
    else:
        logger.info(
            "Preserving explicit max_position_embeddings=%d (seq_length=%d)",
            explicit_max_pos, args.seq_length,
        )

    env_orig_max_pos = os.environ.get("ORIG_MAX_POS_EMB")
    if env_orig_max_pos:
        args.original_max_position_embeddings = int(env_orig_max_pos)
        logger.info(
            "Setting original_max_position_embeddings=%d from ORIG_MAX_POS_EMB env var",
            args.original_max_position_embeddings,
        )
    elif not getattr(args, "original_max_position_embeddings", None):
        if explicit_max_pos and explicit_max_pos > args.seq_length:
            args.original_max_position_embeddings = explicit_max_pos
            logger.info(
                "Setting original_max_position_embeddings=%d from explicit "
                "--max-position-embeddings (seq_length=%d)",
                explicit_max_pos, args.seq_length,
            )

    # TODO: revisit this when megatron(dev) have solved the optimizer-cpu-offload ckpt saving bug
    args.dist_ckpt_save_pre_mcore_014 = True
    # compatible for megatron
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    if not args.tokenizer_model and not args.tokenizer_type:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"
    return args
