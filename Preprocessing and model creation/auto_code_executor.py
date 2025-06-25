import traceback
from llm_helper import LLMWrapper
import os
import re
class AutoCodeExecutor:
    def __init__(self, code: str, max_attempts: int = 5):
        self.original_code = code
        self.code = code
        self.max_attempts = max_attempts
        self.llm = LLMWrapper(temperature=0.2).get_llm()

    def try_execute(self):
        attempt = 0

        while attempt < self.max_attempts:
            try:
                print(f"🛠 Attempt {attempt + 1}...")
                print("🔎 Current Code Preview:\n", self.code[:300])  # Preview first 300 chars

                self.code = self.clean_code(self.code)  # clean each time
                compiled_code = compile(self.code, "<string>", "exec")
                exec(compiled_code, {})
                
                print("✅ Code executed successfully!")
                return True  # Success
            except Exception as e:
                print(f"❌ Error detected: {e}")
                error_message = traceback.format_exc()

                # Ask LLM to fix the code
                self.code = self.ask_llm_to_fix_code(self.code, error_message)

                attempt += 1
        
        print("❌ Failed to fix code after maximum attempts.")
        return False  # Failed after retries


    def clean_code(self, text):
        text = text.strip()
        text = re.sub(r"^```(python)?\n", "", text)
        text = re.sub(r"^```", "", text)
        text = re.sub(r"```$", "", text)

        first_line = text.split("\n")[0]              # ← fixed name
        if not (
            first_line.startswith(("import", "from", "def", "class"))
        ):
            text = "\n".join(text.split("\n")[1:])

        return text


    def simplify_error(self,error_message) : 
        return error_message.strip().split("\n")[-1]


    def ask_llm_to_fix_code(self, code: str, error_message: str) -> str:
        simple_error = self.simplify_error(error_message)
        
        fix_prompt = f"""
You are a Python expert.

### Code:
{code}

### Error:
{simple_error}

Fix this Python script by resolving the error above.

- Generate a fully corrected Python script.
- Do NOT explain or describe anything.
- Output ONLY fixed Python code.
Start immediately with valid Python code.
"""

        fixed_code = self.llm.invoke(fix_prompt)

        # If LLM response wrapped inside "content"
        if hasattr(fixed_code, "content"):
            fixed_code = fixed_code.content

        print("🔧 LLM generated a fixed version!")
        return fixed_code.strip()


    def save_fixed_code(self, filepath="generated_code/fixed_pipeline.py"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(self.code)
        print(f"✅ Saved fixed code to: {filepath}")

    def ask_llm_to_fix_preprocess_function(self, code: str, error_message: str, 
                                       raw_code: str, 
                                       module_import: str = "generated_code.merged_pipeline") -> str:
        """
        Fix a broken `preprocess(df)` function using helper functions from raw_code,
        ensuring imports are used instead of copying full definitions.

        Args:
            code: The broken code to fix.
            error_message: The exception that was raised.
            raw_code: Full source code of helper functions (e.g., from merged_pipeline.py).
            module_import: Module to import functions from.

        Returns:
            The fixed `preprocess(df)` code string.
        """
        simple_error = self.simplify_error(error_message)

        # Step 1: Extract helper function names
        extract_prompt = f"""
    The code below defines several helper functions used in data preprocessing.
    Extract and return a comma-separated list of function names only.

    Code:
    {raw_code}
    """

        extracted = self.llm.invoke(extract_prompt)
        if hasattr(extracted, "content"):
            extracted = extracted.content.strip()

        function_names = extracted.replace("\n", "").strip()  # clean output like: "scale, add_noise, encode_labels"

        # Step 2: Fix the broken code
        fix_prompt = f"""
    Fix the following broken Python function `preprocess(df)`, which failed with this error:

    {simple_error}

    If needed, import the following:
    from {module_import} import {function_names}

    Only return the corrected Python code. No comments. No explanations.

    Broken code:
    {code}
    """

        fixed_code = self.llm.invoke(fix_prompt)
        if hasattr(fixed_code, "content"):
            fixed_code = fixed_code.content

        print("🔁 LLM fixed preprocess function using external imports.")
        return fixed_code.strip()

    def try_execute_preprocess_function(self, raw_code_path: str, module_import: str = "generated_code.merged_pipeline") -> bool:
        """
        Attempt to execute the current `preprocess(df)` code.
        If it fails, ask the LLM to fix it using external helper functions defined in a raw code file.

        Args:
            raw_code_path: Path to the original merged pipeline file for context.
            module_import: Python import path (default: generated_code.merged_pipeline)

        Returns:
            True if successful, False after max retries.
        """
        attempt = 0

        # Load raw pipeline source code once
        try:
            with open(raw_code_path, "r", encoding="utf-8") as f:
                raw_code = f.read()
        except Exception as e:
            print(f"❌ Failed to read raw code: {e}")
            return False

        while attempt < self.max_attempts:
            try:
                print(f"🛠 Attempt {attempt + 1}...")
                print("🔎 Current Code Preview:\n", self.code[:300])  # Preview first 300 chars

                self.code = self.clean_code(self.code)
                compiled_code = compile(self.code, "<string>", "exec")
                exec(compiled_code, {})

                print("✅ preprocess(df) executed successfully!")
                return True
            except Exception as e:
                print(f"❌ Error detected: {e}")
                error_message = traceback.format_exc()

                # Ask LLM to fix using specialized logic
                self.code = self.ask_llm_to_fix_preprocess_function(
                    code=self.code,
                    error_message=error_message,
                    raw_code=raw_code,
                    module_import=module_import
                )

                attempt += 1

        print("❌ Failed to fix preprocess function after maximum attempts.")
        return False
