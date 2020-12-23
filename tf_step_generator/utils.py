from typing import Any, Optional, Tuple
import torch
from torch.nn import functional as F
from transformers import LogitsProcessorList, \
    TopPLogitsWarper, TopKLogitsWarper, NoRepeatNGramLogitsProcessor

def default_get_next_token_logits(model_output: Any) -> torch.Tensor:
    logits: torch.FloatTensor = model_output.logits

    return logits[:, -1, :]


def get_logits_processor_list(
    top_k: Optional[int],
    top_p: Optional[float],
    no_repeat_ngram_size: Optional[int]
) -> LogitsProcessorList:
    items = LogitsProcessorList()

    if no_repeat_ngram_size is not None:
        items.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if top_p is not None:
        items.append(TopPLogitsWarper(top_p))

    if top_k is not None:
        items.append(TopKLogitsWarper(top_k))

    return items

def get_token_ids_and_probabilities(
    logits: torch.FloatTensor
) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    probs = F.softmax(logits, dim=-1)
    probs = probs.squeeze(0)

    next_tokens_ids = torch.nonzero(probs).squeeze(-1)
    next_tokens_probs = probs[probs > 0]

    return next_tokens_ids, next_tokens_probs # type: ignore