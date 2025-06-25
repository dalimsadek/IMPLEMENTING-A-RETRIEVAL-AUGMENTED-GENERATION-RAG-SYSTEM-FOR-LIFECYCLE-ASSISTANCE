# tools.py
# ---------------------------------------------------------------------------
# High-level helper-tools for the agent loop
# ---------------------------------------------------------------------------
from __future__ import annotations
from model_trainer_generator import ModelTrainerCodeGenerator
import inspect
import importlib.util
import json, pathlib, pandas as pd
from pathlib import Path
from typing import Dict, List
import importlib.util, sys, os

from langchain_core.tools import tool

from context_builder    import ContextBuilder
from data_preprocessor  import DataPreprocessor, PreprocessingCodeGenerator
from merger             import CodeMergerAndFixer
from auto_code_executor import AutoCodeExecutor


# ---------------------------------------------------------------------------
# 1) SHARED STATE ACROSS TOOL-CALLS
# ---------------------------------------------------------------------------
CTX: Dict = {          # global “memory” for the agent
    "context"      : None,   # dataset-level context   (dict)
    "suggestions"  : None,   # {task_name: description}
    "chunks"       : None,   # List[Document] from retriever
    "merged_path"  : None,   # str | None   –  last merged script
}
GENERATED: Dict[str, str] = {}          # task_name -> code-snippet text


# ---------------------------------------------------------------------------
# 2)  load_context
# ---------------------------------------------------------------------------
@tool
def load_context(csv_path:str,
                 task_description:str,
                 feature_desc_json:str,
                 label_desc:str,
                 notes:str="") -> str:
    """
    Build a context object from a CSV and return a human-readable summary.

    • `feature_desc_json` must be a JSON string:
        {"col_name":"short description", ... }
    """
    builder = ContextBuilder(
        csv_path           = csv_path,
        task_description   = task_description,
        feature_description= json.loads(feature_desc_json),
        label_description  = label_desc,
        notes              = notes
    )
    CTX["context"] = builder.get_context()
    # reset everything else when a new dataset is loaded
    CTX["suggestions"] = None
    CTX["chunks"]      = None
    CTX["merged_path"] = None
    GENERATED.clear()
    return builder.get_context_as_text()


# ---------------------------------------------------------------------------
# 3)  suggest_preprocessing
# ---------------------------------------------------------------------------
@tool
def suggest_preprocessing(top_k:int=4) -> str:
    """
    Ask the LLM (plus relevant Docs) for a dictionary of preprocessing tasks.
    Returns that dictionary as pretty-printed JSON.
    """
    if CTX["context"] is None:
        raise RuntimeError("You must `load_context` first.")

    pre  = DataPreprocessor(context=CTX["context"], top_k=top_k)
    sugg = pre.suggest_tasks()                  # dict(task -> description)

    CTX["suggestions"] = sugg
    CTX["chunks"]      = pre.retriever.get_relevant_documents(
                            f"{CTX['context']['task']} {CTX['context']['notes']}"
                        )
    return json.dumps(sugg, indent=2)


# ---------------------------------------------------------------------------
# 4)  generate_code
# ---------------------------------------------------------------------------
@tool
def generate_code(task_name: str) -> str:
    """
    Generate python code for one preprocessing task and save in memory.
    """
    gen   = PreprocessingCodeGenerator(
                context          = CTX["context"],
                task_suggestions = CTX["suggestions"],
                chunks           = CTX["chunks"])

    raw   = gen.generate_code(task_name, CTX["suggestions"][task_name])
    code  = str(raw)           # <-- unwrap to str

    GENERATED[task_name] = code    # keep the full snippet

    preview = code[:220] + (" …" if len(code) > 220 else "")
    return f"[{task_name}] snippet stored ({len(code)} chars):\n{preview}"


# ---------------------------------------------------------------------------
# 5)  merge_snippets  (NEW)
# ---------------------------------------------------------------------------
@tool
def merge_snippets() -> str:
    """
    Merge **all** snippets collected so far (in insertion order) into one clean
    script using CodeMergerAndFixer.  The merged file is saved on disk and its
    path is returned.  Does *NOT* execute the code.
    """
    if CTX["context"] is None:
        raise RuntimeError("Context not loaded.")
    if not GENERATED:
        raise RuntimeError("No snippets to merge.  Call generate_code first.")

    ordered = list(GENERATED.keys())
    fixer   = CodeMergerAndFixer(
                  context = CTX["context"],
                  tasks   = {t: GENERATED[t] for t in ordered}
             )
    merged_code = fixer.merge_with_llm()

    out_path = Path("generated_code/merged_pipeline.py")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(merged_code, encoding="utf-8")

    CTX["merged_path"] = str(out_path)          # remember
    return f"Merged {len(ordered)} snippets → {out_path}"



# ---------------------------------------------------------------------------
# ---------- 5) run_pipeline_from_file ----------
@tool
def run_pipeline_from_file() -> str:
    """
    Compile & execute the *already* merged script saved at
    generated_code/merged_pipeline.py.
    Returns ✅/❌ with the first traceback line if it fails.
    """
    from auto_code_executor import AutoCodeExecutor
    import pathlib, traceback

    script_path = pathlib.Path("generated_code/merged_pipeline.py")
    if not script_path.exists():
        return "❌ merged script not found. Run merge_snippets() first."

    code = script_path.read_text()
    exec_ = AutoCodeExecutor(code, max_attempts=7)
    ok    = exec_.try_execute()
    return "✅ pipeline executed" if ok else "❌ pipeline failed – see pipeline_failed.py"

# ─────────────────────────────────────────────────────────────────────────────
# 6️⃣  build_preprocess_module   (new tool)
# ─────────────────────────────────────────────────────────────────────────────
from data_preprocessor import PreprocessFunctionBuilder
from langchain_core.tools import tool


@tool
def build_preprocess_module(pipeline_path: str, csv_sample: str | None = None) -> str:
    """
    Convert a monolithic pipeline script into a `preprocess(df)` module,
    and attempt to execute it with AutoCodeExecutor.

    Args:
        pipeline_path: path to the script produced by the LLM (e.g., merged_pipeline.py)
        csv_sample   : optional small csv file to call the function on

    Returns:
        "✅ <file>" if success (generation + execution), otherwise "❌ <reason>"
    """
    output_path = "generated_code/preprocess_module.py"

    # Step 1: Build initial preprocess module
    builder = PreprocessFunctionBuilder(
        src_path=pipeline_path,
        dst_path=output_path,
        csv_sample=csv_sample
    )
    
    ok, msg = builder.build()
    if not ok:
        return f"❌ Failed to build module: {msg}"

    # Step 2: Read the generated code
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            generated_code = f.read()
    except Exception as e:
        return f"❌ Failed to read generated code: {e}"

    # Step 3: Run personalized preprocess function executor
    executor = AutoCodeExecutor(code=generated_code, max_attempts=10)
    success = executor.try_execute_preprocess_function(
        raw_code_path=pipeline_path,  # Use original file for import context
        module_import="generated_code.merged_pipeline"
    )

    # Step 4: Save final fixed version
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(executor.code)
    except Exception as e:
        return f"❌ Execution OK, but failed to save corrected code: {e}"

    if success:
        return "✅ preprocess_module.py generated and executed successfully!"
    else:
        return "❌ Code generation failed after auto-fixing attempts. Final version saved."


@tool
def generate_training_code(preprocess_module_path: str) -> str:
    """
    Generate and execute deep learning model code using Keras based on the preprocess(df) function
    and dataset context. Assumes X_train, y_train will be used for fitting and saves model as .h5.

    Args:
        preprocess_module_path: path to module with preprocess(df)

    Returns:
        Status message indicating success or failure.
    """
    if CTX["context"] is None:
        raise RuntimeError("You must `load_context` first.")

    output_path = "generated_code/train_model.py"
    model_save_path = "models/final_model.h5"

    # Step 1: Import the preprocess function
    try:
        spec = importlib.util.spec_from_file_location("preprocess_module", preprocess_module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        preprocess_code = inspect.getsource(module.preprocess)
    except Exception as e:
        return f"❌ Failed to import preprocess function: {e}"

    # Step 2: Generate the training code
    generator = ModelTrainerCodeGenerator(context=CTX["context"], top_k=5)
    code = generator.generate_training_code(preprocess_code)
    if hasattr(code, "content"):
        code = code.content

    # Step 3: Append model saving step (if not already included)
    if "model.save(" not in code:
        code += f"\n\n# Save trained model\nimport os\nos.makedirs('models', exist_ok=True)\nmodel.save('{model_save_path}')\n"

    # Step 4: Try executing it using AutoCodeExecutor
    executor = AutoCodeExecutor(code=code, max_attempts=10)
    success = executor.try_execute()

    # Step 5: Save the final (possibly fixed) version
    Path("generated_code").mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(executor.code, encoding="utf-8")

    # Step 6: Return result
    if success:
        return f"✅ Training code executed and model saved to {model_save_path}!"
    else:
        return "❌ Training code failed after auto-fix attempts. Final version saved."

from llm_helper import LLMWrapper

@tool
def convert_to_tflite_model(original_model_path: str, 
                            converted_model_path: str, 
                            input_datatype: str = "float32", 
                            output_datatype: str = "float32",
                            quantization: bool = False) -> str:
    """
    Convert a trained Keras model to TFLite format. Supports optional quantization.

    Args:
        original_model_path: Path to the .h5 or .keras model
        converted_model_path: Destination path for the TFLite model
        input_datatype: Inference input type (float32, int8)
        output_datatype: Inference output type
        quantization: Whether to apply quantization

    Returns:
        Success or failure message.
    """
    quant_desc = " with quantization" if quantization else " without quantization"

    # === PROMPT: Initial Conversion ===
    prompt = f"""
You are an expert TinyML and TensorFlow engineer.

Your task is to generate executable Python code that converts a trained Keras model into TFLite format.

### Requirements
- The model is located at: {original_model_path}
- The output TFLite file must be saved to: {converted_model_path}
- Input data type: {input_datatype}
- Output data type: {output_datatype}
- {"Quantization must be applied" if quantization else "Do not apply quantization unless explicitly required"}

### Implementation Instructions
- Always use `model.save('path', save_format='tf')` if saving a model.
- If loading from `.h5`, pass `compile=False` in `load_model()`.
- Avoid using `InputLayer` with `batch_shape`. Use `input_shape=(X_train.shape[1],)` if needed.
- Prefer conversion from SavedModel using: `tf.lite.TFLiteConverter.from_saved_model(...)`
- If `.h5` must be used, handle deserialization errors gracefully.
- Include fallback logic using `get_concrete_function()` if the direct conversion fails.
- After defining the converter, you must call `converter.convert()` and assign it to `tflite_model`
- Save the converted model using:
  `with open('{converted_model_path}', 'wb') as f: f.write(tflite_model)`
- Output only the corrected Python code. No markdown, no explanation, no extra text.

Start now.
"""



    llm = LLMWrapper(temperature=0.3).get_llm()
    code = llm.invoke(prompt)
    if hasattr(code, "content"):
        code = code.content.strip()

    # === Step 2: Execute the code ===
    executor = AutoCodeExecutor(code=code, max_attempts=3)
    success = executor.try_execute()

    if not success:
        # === PROMPT: Error Recovery ===
        error_info = executor.last_error if hasattr(executor, "last_error") else "Unknown"
        error_prompt = f"""
You are a TinyML engineer. Regenerate Python code that converts a Keras model to TFLite format{quant_desc}.

Original model path: {original_model_path}
Converted model path: {converted_model_path}
Input type: {input_datatype}
Output type: {output_datatype}

This is the code that failed:
{executor.code}

And this was the error:
{error_info}

Fix the issue and return only the corrected Python code. No explanations or comments.
"""
        retry_code = llm.invoke(error_prompt)
        if hasattr(retry_code, "content"):
            retry_code = retry_code.content.strip()
        executor.code = retry_code
        success = executor.try_execute()

    # === Step 3: Save the final version ===
    Path("generated_code").mkdir(parents=True, exist_ok=True)
    Path("generated_code/convert_model.py").write_text(executor.code, encoding="utf-8")

    return (
        f"✅ Model converted to TFLite and saved at {converted_model_path}."
        if success else
        "❌ Conversion failed after all attempts. Final version saved to convert_model.py."
    )

# ---------------------------------------------------------------------------
# End of tools.py
# ---------------------------------------------------------------------------
