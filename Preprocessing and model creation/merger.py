import re
from llm_helper import LLMWrapper

class CodeMergerAndFixer:
    def __init__(self,context: dict, tasks: dict):
        """
        tasks: dict where key=task_name, value=generated code (string)
        """
        self.context = context
        self.csv_path = context.get("source_file", "data.csv")
        self.tasks = tasks
        self.llm = LLMWrapper(temperature=0.2).get_llm()
        self.full_code = ""

    def merge_with_llm(self) -> str:
        prompt = self._build_prompt()
        merged_code = self.llm.invoke(prompt)

        # Handle result if it comes with content attribute
        if hasattr(merged_code, "content"):
            merged_code = merged_code.content

        self.full_code = self._clean_code_block(merged_code)
        return self.full_code




    def _build_prompt(self) -> str:
        merged_snippets = []

        for task_name, code_snippet in self.tasks.items():
            clean_code = self._clean_code_block(code_snippet)
            merged_snippets.append(f"# === {task_name.upper()} ===\n{clean_code}")

        full_snippets = "\n\n".join(merged_snippets)

        prompt = f"""
You are a Python data engineering expert.

You are given multiple small Python code snippets:

{full_snippets}

Your strict instructions:
- Merge all snippets into one clean Python script.
- Start immediately by importing necessary libraries.
- Load the dataset using:
  df = pd.read_csv('{self.context["source_file"]}')
- Do NOT include any print() statements or debug outputs.
- Do NOT add any comments, markdown formatting, explanations, or descriptive text.
- Deduplicate all imports (e.g., import pandas only once).
- Fix inconsistent or conflicting variable names if necessary.
- Apply all preprocessing steps first.
- Split the dataset into training and testing sets ONLY at the very end, after preprocessing and feature engineering is complete.
- Ensure that the final code runs top-to-bottom without errors.
⚡ Strict Instructions:
- Do NOT say anything like "Here is the code", "Below is the script", or any other introduction.
- Output ONLY clean Python code.
- Start immediately with the Python imports and loading the dataset.

Your output must be pure executable code, and NOTHING else.
"""


        return prompt

    def _clean_code_block(self, code: str) -> str:
        code = code.strip()
        code = re.sub(r"^```[a-z]*\n?", "", code, flags=re.MULTILINE)  # remove ```python
        code = re.sub(r"\n?```$", "", code, flags=re.MULTILINE)        # remove ending ```
        return code

    def save_to_file(self, filename="merged_preprocessing_pipeline.py"):
        if self.full_code:
            with open(filename, "w") as f:
                f.write(self.full_code)
