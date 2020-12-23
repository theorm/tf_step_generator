from typing import Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.cuda
from transformers import PreTrainedModel, PreTrainedTokenizer, BatchEncoding
from tf_step_generator.utils import default_get_next_token_logits, get_logits_processor_list, get_token_ids_and_probabilities

@dataclass
class GenerationStepResults:
    next_ids: torch.LongTensor
    next_probabilities: torch.FloatTensor

    def __init__(self,
                 next_ids: torch.LongTensor,
                 next_probabilities: torch.FloatTensor,
                 tokenizer: PreTrainedTokenizer):
        self.next_ids = next_ids
        self.next_probabilities = next_probabilities
        self.tokenizer = tokenizer

    @property
    def tokens_tuple(self):
        return [
            (self.tokenizer.decode([tid]), prob.item())
            for (tid, prob) in zip(self.next_ids, self.next_probabilities)
        ]

    def sample_token_id(self):
        idx = torch.multinomial(self.next_probabilities, num_samples=1)
        return self.next_ids[idx].item()


class TokenGeneratorBase:
    def __init__(self,
                model: PreTrainedModel,
                tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_step(
        self,
        tokenized_input: BatchEncoding,
        top_k: Optional[int] = 3,
        top_p: Optional[float] = None
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        model_kwargs = {
            **tokenized_input,
            'use_cache': False
        }
        
        with torch.no_grad():
            result = self.model(**model_kwargs)

        logits = default_get_next_token_logits(result)
        logits_processors = get_logits_processor_list(top_k, top_p)
        processed_logits = logits_processors(model_kwargs['input_ids'], logits)
        next_tokens_ids, next_tokens_probabilities = get_token_ids_and_probabilities(processed_logits)
        
        return next_tokens_ids, next_tokens_probabilities

class NextTokenGenerator(TokenGeneratorBase):
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 tokenized_input: BatchEncoding,
                 prompt_length: int):
        super().__init__(model, tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.tokenized_input = tokenized_input
        self.prompt_length = prompt_length

    def __call__(self,
                 next_token_id: int,
                 top_k: Optional[int] = 3,
                 top_p: Optional[float] = None) -> Tuple[GenerationStepResults, 'NextTokenGenerator']:

        self.tokenized_input['input_ids'] = torch.cat([
            self.tokenized_input['input_ids'],
            torch.LongTensor([[next_token_id]]).to(self.model.device),
        ], dim=-1)

        if 'attention_mask' in self.tokenized_input:
            self.tokenized_input['attention_mask'] = torch.cat([
                self.tokenized_input['attention_mask'],
                torch.LongTensor([[1]]).to(self.model.device),
            ], dim=-1)

        next_tokens_ids, next_tokens_probabilities = self.generate_step(
            self.tokenized_input,
            top_k=top_k,
            top_p=top_p
        )

        step_results = GenerationStepResults(
            next_tokens_ids,
            next_tokens_probabilities,
            self.tokenizer
        )

        return step_results, self

    def get_generated_text(self) -> str:
        return self.tokenizer.decode(self.tokenized_input['input_ids']
            .squeeze(0)[self.prompt_length:])

    def get_prompt_text(self) -> str:
        return self.tokenizer.decode(self.tokenized_input['input_ids']
            .squeeze(0)[:self.prompt_length])


class FirstTokenGenerator(TokenGeneratorBase):
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer):
        super().__init__(model, tokenizer)


    def __call__(self,
                 input_text: str,
                 top_k: Optional[int] = 3,
                 top_p: Optional[float] = None) -> Tuple[GenerationStepResults, NextTokenGenerator]:
        tokenized_input = self.tokenizer(input_text, return_tensors='pt')
        tokenized_input = {k: v.to(self.model.device) for k, v in tokenized_input.items()}

        prompt_length = tokenized_input['input_ids'].shape[-1]

        next_tokens_ids, next_tokens_probabilities = self.generate_step(
            tokenized_input,
            top_k=top_k,
            top_p=top_p
        )

        step_results = GenerationStepResults(
            next_tokens_ids,
            next_tokens_probabilities,
            self.tokenizer
        )
        next_token_generator = NextTokenGenerator(
            self.model,
            self.tokenizer,
            tokenized_input,
            prompt_length=prompt_length
        )

        return step_results, next_token_generator
    