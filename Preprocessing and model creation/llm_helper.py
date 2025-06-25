# llm_instance.py

import os
from langchain_groq import ChatGroq

class LLMWrapper:
    def __init__(
        self,
        model_name: str = "llama3-70b-8192",
        temperature: float = 0.0,
        api_key: str = None,
        verbose: bool = True
    ):
        """
        Initialize a Groq LLM with customizable settings.

        Args:
            model_name (str): Name of the Groq model to use.
            temperature (float): Sampling temperature for output diversity.
            api_key (str): Groq API key (uses env var if None).
            verbose (bool): Whether to print init confirmation.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing. Set it via env or parameter.")

        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature
        )

        if verbose:
            print(f"[LLMWrapper] Initialized Groq LLM: {self.model_name} (temp={self.temperature})")

    def get_llm(self):
        """Return the internal LLM instance."""
        return self.llm
