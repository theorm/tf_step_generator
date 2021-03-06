from typing import Optional, Tuple, Any, Callable
from dataclasses import dataclass
import torch
import torch.cuda
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.file_utils import ModelOutput
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
                tokenizer: PreTrainedTokenizer,
                prepare_model_inputs: Callable[[Any], Any] = lambda x: x,
                get_next_token_logits: Callable[[Any], torch.Tensor] = default_get_next_token_logits):
        self.model = model
        self.tokenizer = tokenizer
        self.prepare_model_inputs = prepare_model_inputs
        self.get_next_token_logits = get_next_token_logits

    def generate_step(
        self,
        model_kwargs: Any,
        top_k: Optional[int] = 3,
        top_p: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, Any]:
        input_ids = model_kwargs['input_ids']
        del model_kwargs['input_ids']
        model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        model_inputs = self.prepare_model_inputs(model_inputs)

        with torch.no_grad():
            result = self.model(**model_inputs, return_dict=True)

        logits = self.get_next_token_logits(result)
        logits_processors = get_logits_processor_list(top_k, top_p, no_repeat_ngram_size=no_repeat_ngram_size)
        processed_logits = logits_processors(input_ids, logits)
        next_tokens_ids, next_tokens_probabilities = get_token_ids_and_probabilities(processed_logits)

        model_kwargs = self.model._update_model_kwargs_for_generation(
            result, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
        )

        model_kwargs['input_ids'] = input_ids

        return next_tokens_ids, next_tokens_probabilities, model_kwargs

class NextTokenGenerator(TokenGeneratorBase):
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 model_kwargs: Any,
                 prompt_length: int,
                 prepare_model_inputs: Callable[[Any], Any] = lambda x: x,
                 get_next_token_logits: Callable[[Any], torch.Tensor] = default_get_next_token_logits,
                 top_k: Optional[int] = 3,
                 top_p: Optional[float] = None,
                 no_repeat_ngram_size: Optional[int] = None):
        super().__init__(model, tokenizer, prepare_model_inputs, get_next_token_logits)
        self.model = model
        self.tokenizer = tokenizer
        self.model_kwargs = model_kwargs
        self.prompt_length = prompt_length

        self.previous_top_k = top_k
        self.previous_top_p = top_p
        self.previous_no_repeat_ngram_size = no_repeat_ngram_size

    def __call__(self,
                 next_token_id: int,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 no_repeat_ngram_size: Optional[int] = None) -> Tuple[GenerationStepResults, 'NextTokenGenerator']:

        self.model_kwargs['input_ids'] = torch.cat([
            self.model_kwargs['input_ids'],
            torch.LongTensor([[next_token_id]]).to(self.model.device),
        ], dim=-1)

        next_tokens_ids, next_tokens_probabilities, model_kwargs = self.generate_step(
            self.model_kwargs,
            top_k=top_k if top_k is not None else self.previous_top_k,
            top_p=top_p if top_p is not None else self.previous_top_p,
            no_repeat_ngram_size=no_repeat_ngram_size if no_repeat_ngram_size is not None else self.previous_no_repeat_ngram_size
        )
        self.model_kwargs = model_kwargs

        step_results = GenerationStepResults(
            next_tokens_ids,
            next_tokens_probabilities,
            self.tokenizer
        )

        return step_results, self

    def get_generated_text(self) -> str:
        offset = 0 if self.model.config.is_encoder_decoder else self.prompt_length
        return self.tokenizer.decode(self.model_kwargs['input_ids']
            .squeeze(0)[offset:], skip_special_tokens=True)

class FirstTokenGenerator(TokenGeneratorBase):
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 prepare_model_inputs: Callable[[Any], Any] = lambda x: x,
                 prepare_tokenized_input: Callable[[Any], Any] = lambda x: x,
                 get_next_token_logits: Callable[[Any], torch.Tensor] = default_get_next_token_logits):
        super().__init__(model, tokenizer, prepare_model_inputs, get_next_token_logits=get_next_token_logits)
        self.prepare_tokenized_input = prepare_tokenized_input

    def __call__(self,
                 input_text: str,
                 top_k: Optional[int] = 3,
                 top_p: Optional[float] = None,
                 no_repeat_ngram_size: Optional[int] = None) -> Tuple[GenerationStepResults, NextTokenGenerator]:
        tokenized_input = self.tokenizer(input_text, return_tensors='pt')
        tokenized_input = self.prepare_tokenized_input(tokenized_input)
        tokenized_input = {k: v.to(self.model.device) for k, v in tokenized_input.items()}

        model_kwargs = {
            **tokenized_input
        }

        # From: https://github.com/huggingface/transformers/blob/1558d191e66fe3b5b34c8d9a6ce657a39d5133ae/src/transformers/generation_utils.py#L501
        if self.model.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            input_ids = model_kwargs['input_ids']
            del model_kwargs['input_ids']
            model_kwargs = self.model._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)

            # NOTE: this is originally passed as an argument
            decoder_start_token_id = None 
            
            bos_token_id = self.model.config.bos_token_id
            # set input_ids as decoder_input_ids
            model_kwargs['input_ids'] = self.model._prepare_decoder_input_ids_for_generation(
                input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id, **model_kwargs
            )

            if "encoder_outputs" not in model_kwargs or not isinstance(model_kwargs["encoder_outputs"], ModelOutput):
                raise ValueError("Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.")

        # From: https://github.com/huggingface/transformers/blob/1558d191e66fe3b5b34c8d9a6ce657a39d5133ae/src/transformers/generation_utils.py#L564
        num_return_sequences = 1

        input_ids = model_kwargs['input_ids']
        del model_kwargs['input_ids']

        input_ids, model_kwargs = self.model._expand_inputs_for_generation(
            input_ids,
            expand_size=num_return_sequences,
            is_encoder_decoder=self.model.config.is_encoder_decoder,
            **model_kwargs,
        )
        model_kwargs['input_ids'] = input_ids

        prompt_length = tokenized_input['input_ids'].shape[-1]

        model_kwargs['use_cache'] = False

        next_tokens_ids, next_tokens_probabilities, model_kwargs = self.generate_step(
            model_kwargs,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size
        )

        step_results = GenerationStepResults(
            next_tokens_ids,
            next_tokens_probabilities,
            self.tokenizer
        )
        next_token_generator = NextTokenGenerator(
            self.model,
            self.tokenizer,
            model_kwargs,
            prompt_length=prompt_length,
            prepare_model_inputs=self.prepare_model_inputs,
            get_next_token_logits=self.get_next_token_logits,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size
        )

        return step_results, next_token_generator
    