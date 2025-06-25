from retriever_instance import RetrieverBuilder
from llm_helper import LLMWrapper
import inspect

class ModelTrainerCodeGenerator:
    def __init__(self, context: dict, top_k: int = 5):
        self.context = context
        self.llm = LLMWrapper(temperature=0.3).get_llm()
        self.retriever = RetrieverBuilder(top_k=top_k).get_retriever()
        self.document_context = self._retrieve_chunks()

    def _retrieve_chunks(self) -> str:
        query = f"train deep learning model for {self.context['task']}"
        docs = self.retriever.get_relevant_documents(query)
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_training_code(self, preprocess_code: str) -> str:
        features = ", ".join(self.context["features"])
        prompt = f"""
You are a deep learning engineer.

You are given a preprocessing function and dataset context.
Write executable Python code using Keras to build and train a deep learning model.

- Do NOT redefine X_train or y_train.
- Import them from `generated_code.merged_pipeline`.
- Assume they were produced after calling `preprocess(df)` on the original dataset.
- Build a model suitable for binary classification.
- Set the input shape dynamically using: input_shape=(X_train.shape[1],)
- Do not hardcode any feature dimension numbers.

### Preprocess Function (for reference only)
{preprocess_code}

### Dataset Context
- Task: {self.context['task']}
- Features: {features}
- Label: {self.context['label_explanation']}
- Notes: {self.context['notes']}

### Retrieved Documentation
{self.document_context}

Only return valid Python code.
Start with:
```python
from generated_code.merged_pipeline import X_train, y_train
Do NOT include comments, markdown, or explanations.
"""
        return self.llm.invoke(prompt)

