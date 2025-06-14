from config import device
from langchain.llms.base import LLM
from typing import Optional, List
import torch

class SimpleHuggingFaceLLM(LLM):
    model: any
    tokenizer: any
    device: str = device

    @property
    def _llm_type(self) -> str:
        return "simple_huggingface"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=300,
            do_sample=False,
            temperature=0.0,
            top_k=None,
            top_p=None
        )

        output_ids = outputs[0][inputs.input_ids.shape[-1]:]
        decoded_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return decoded_output.strip()