"""BPE tokenizer shared by parts 1 and 2.

Byte-level BPE (GPT-2 style) so that non-romanized Vietnamese/Japanese text is
handled as raw UTF-8 bytes without any language-specific preprocessing.
"""

from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def train_bpe_tokenizer(
    texts: Iterable[str],
    vocab_size: int,
    save_path: Path,
) -> PreTrainedTokenizerFast:
    """Train a byte-level BPE tokenizer on an iterable of raw strings."""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer)

    pad_id = tokenizer.token_to_id("<pad>")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))

    return PreTrainedTokenizerFast(
        tokenizer_file=str(save_path),
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


def train_tokenizer_from_dataset(
    dataset,
    columns: Sequence[str],
    vocab_size: int,
    save_path: Path,
) -> PreTrainedTokenizerFast:
    """Train a tokenizer on the given text columns of a HuggingFace dataset."""

    def gen() -> Iterable[str]:
        for row in dataset:
            for col in columns:
                yield row[col]

    return train_bpe_tokenizer(gen(), vocab_size, save_path)


def load_tokenizer(path: Path) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path),
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )