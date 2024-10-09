from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class HindiTranslator:
    def __init__(self):
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")
    
    def to_hindi(self, text: str) -> str:
        model_inputs = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **model_inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id["hi_IN"]
        )
        translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return translation[0]
    
    def to_telugu(self, text: str) -> str:
        model_inputs = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **model_inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id["te_IN"]
        )
        translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return translation[0]
    
    def to_tamil(self, text: str) -> str:
        model_inputs = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **model_inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id["ta_IN"]
        )
        translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return translation[0]

# Example usage:


