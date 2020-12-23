import unittest
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tf_step_generator.generator import FirstTokenGenerator

MODEL_NAME = 'distilgpt2'

class GenerateTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        cls.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    def test_generate_one_simple(self):
        generator = FirstTokenGenerator(self.model, self.tokenizer)
        step, next_token_generator = generator('In a galaxy', top_k=3)
        assert step.tokens_tuple == [
            ('-', 0.10171877592802048), 
            (' far', 0.787763237953186), 
            (' full', 0.1105179414153099)
        ]
        assert next_token_generator is not None
        assert step.sample_token_id() is not None
        assert next_token_generator.get_generated_text() == ''
        assert next_token_generator.get_prompt_text() == 'In a galaxy'

    def test_generate_two_tokens(self):
        generator = FirstTokenGenerator(self.model, self.tokenizer)
        step, next_token_generator = generator('In a galaxy', top_k=3)

        next_token_id = int(step.next_ids[0].item())

        step, next_token_generator = next_token_generator(next_token_id)
        assert step.tokens_tuple == [
            ('wide', 0.6917651891708374),
            ('span', 0.2856826186180115),
            ('sized', 0.022552236914634705)
        ]
        assert next_token_generator is not None
        assert step.sample_token_id() is not None

        assert next_token_generator.get_generated_text() == '-'
        assert next_token_generator.get_prompt_text() == 'In a galaxy'


if __name__ == '__main__':
    unittest.main()
