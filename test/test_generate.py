import unittest
from transformers import GPT2LMHeadModel, GPT2Tokenizer, \
    T5Tokenizer, T5ForConditionalGeneration
from tf_step_generator.generator import FirstTokenGenerator

class GenerateLMTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        MODEL_NAME = 'distilgpt2'
        ModelClass, TokenizerClass = GPT2LMHeadModel, GPT2Tokenizer

        cls.tokenizer = TokenizerClass.from_pretrained(MODEL_NAME)
        cls.model = ModelClass.from_pretrained(MODEL_NAME)

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

class GenerateConditionalTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        MODEL_NAME = 'valhalla/distilt5-qg-hl-6-4'
        TokenizerClass, ModelClass = T5Tokenizer, T5ForConditionalGeneration

        cls.tokenizer = TokenizerClass.from_pretrained(MODEL_NAME)
        cls.model = ModelClass.from_pretrained(MODEL_NAME)

    def test_generate_one_simple(self):
        generator = FirstTokenGenerator(self.model, self.tokenizer)
        step, next_token_generator = generator('In a galaxy', top_k=3)
        assert step.tokens_tuple == [
            ('In', 0.8047278523445129), 
            ('What', 0.07219909876585007), 
            ('Where', 0.12307299673557281)
        ]
        assert next_token_generator is not None
        assert step.sample_token_id() is not None
        assert next_token_generator.get_generated_text() == ''

    def test_generate_two_tokens(self):
        generator = FirstTokenGenerator(self.model, self.tokenizer)
        step, next_token_generator = generator('In a galaxy', top_k=3)

        next_token_id = int(step.next_ids[0].item())

        step, next_token_generator = next_token_generator(next_token_id)
        assert step.tokens_tuple == [
            ('', 0.924037754535675),
            ('the', 0.030165817588567734),
            ('an', 0.045796316117048264)
        ]
        assert next_token_generator is not None
        assert step.sample_token_id() is not None

        assert next_token_generator.get_generated_text() == 'In'


if __name__ == '__main__':
    unittest.main()
