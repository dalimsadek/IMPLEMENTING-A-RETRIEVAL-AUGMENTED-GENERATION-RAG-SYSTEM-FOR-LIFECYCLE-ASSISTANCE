from retriever_instance import RetrieverBuilder
from llm_helper import LLMWrapper
class DataPreprocessor:
    def __init__(self, context: dict, top_k=5):
        self.context = context
        self.llm = LLMWrapper(temperature=0.3).get_llm()
        self.retriever = RetrieverBuilder(top_k=top_k).get_retriever()
        self.document_context = self._retrieve_chunks()

    def _retrieve_chunks(self) -> str:
        query = f"{self.context['task']} {self.context['notes']}"
        docs = self.retriever.get_relevant_documents(query)
        return "\n\n".join(doc.page_content for doc in docs)

    def _context_to_text(self, ctx: dict) -> str:
        lines = [
            f"Task: {ctx['task']}",
            "Features:",
        ] + [
            f"  - {k}: {v}" for k, v in ctx["feature_explanation"].items()
        ] + [
            f"Label Info: {ctx['label_explanation']}",
            f"Notes: {ctx['notes']}",
            "Sample Data:"
        ] + [
            f"  - {row}" for row in ctx["sample_data"]
        ]
        return "\n".join(lines)

    def suggest_tasks(self) -> dict:
        context_text = self._context_to_text(self.context)

        prompt = f"""
    You are a TinyML data engineering assistant.

    Here is the user's dataset context:

    {context_text}

    Below is some documentation retrieved from a technical knowledge base:

    {self.document_context}

    Suggest a list of preprocessing and data engineering tasks in dictionary format:

    {{
    "task_name": "Short explanation of what it does"
    }}

    Only return the dictionary. No explanations, no markdown.
    """

        # ✅ Always assign the result first
        result = self.llm.invoke(prompt)

        # ✅ If it's an AIMessage, get the string content
        if hasattr(result, "content"):
            result = result.content

        # ✅ If it's a string, parse it
        if isinstance(result, str):
            try:
                return eval(result)
            except Exception:
                import json
                return json.loads(result)

        return result

class PreprocessingCodeGenerator:
    def __init__(self, context: dict, task_suggestions: dict, chunks: list):
        self.context = context
        self.task_suggestions = task_suggestions
        self.chunks = chunks  # list of LangChain Document objects
        self.llm = LLMWrapper(temperature=0.3).get_llm()

    def _chunks_to_text(self) -> str:
        return "\n\n".join(chunk.page_content for chunk in self.chunks)

    def generate_code_for_all_tasks(self) -> dict:
        return {
            task: self.generate_code(task, desc)
            for task, desc in self.task_suggestions.items()
        }

    def generate_code(self, task_name: str, task_description: str) -> str:
        chunk_context = self._chunks_to_text()

        prompt = f"""
You are a Python data engineer.

Implement the following preprocessing task:
"{task_name}": {task_description}

Context about the dataset:
- Task: {self.context['task']}
- Features: {', '.join(self.context['features'])}
- Label: {self.context['label_explanation']}
- Notes: {self.context['notes']}

Sample Data (first few rows):
{self.context['sample_data']}

Helpful documentation from external technical sources:
{chunk_context}

Write clean and executable Python code using pandas. The dataset is in a variable named `df`.
Don't generate plots .
Only return valid Python code. Do not include markdown or comments.
"""
        return self.llm.invoke(prompt)
    






class PreprocessFunctionBuilder:
    def __init__(self, src_path: str = "generated_code/preprocess_module.py", dst_path: str = "generated_code/preprocess_module.py", csv_sample: str | None = None):
        self.src_path   = src_path
        self.dst_path   = dst_path
        self.csv_sample = csv_sample
        self.llm        = LLMWrapper(temperature=0.1).get_llm()

    def build(self) -> tuple[bool, str]:
        import os
        if not os.path.exists(self.src_path):
            return False, f"Source file not found: {self.src_path}"

        with open(self.src_path, "r", encoding="utf-8") as f:
            raw_code = f.read()

        prompt = f"""
You are a Python data engineer.

Inspect the code below, which was used to preprocess training data. 
Your task is to generate a clean and minimal function called `preprocess(df)` that applies 
the same preprocessing steps required to make predictions on new, unseen entries.

This includes all feature engineering, scaling, encoding, or transformations done during training.
Do not include data splitting, model training, print statements, or any output text.
STRICT INSTRUCTIONS :   
    - If the `preprocess(df)` function uses any other functions defined in the code 
you must import them using `from generated_code.merged_pipeline import ...`.

    - Return only valid Python code. No comments, no markdown, no explanation — only the function definition and necessary import statement.

Code:
{raw_code}
"""




        result = self.llm.invoke(prompt)

        if hasattr(result, "content"):
            result = result.content

        if isinstance(result, str):
            result = "import pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\nfrom sklearn.model_selection import train_test_split\n\n" + result.strip()

        with open(self.dst_path, "w", encoding="utf-8") as f:
            f.write(result)

        return True, f"Function `preprocess(df)` written to {self.dst_path}"

    def test(self) -> tuple[bool, str]:
        import importlib.util
        import pandas as pd

        spec = importlib.util.spec_from_file_location("preprocess_module", self.dst_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if self.csv_sample is None:
            return True, "No CSV sample provided for test. Skipped execution."

        df = pd.read_csv(self.csv_sample)
        df.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')

        try:
            df_out = module.preprocess(df)
            assert isinstance(df_out, pd.DataFrame)
            return True, "Preprocess function executed successfully."
        except Exception as e:
            return False, f"Execution error: {e}"
