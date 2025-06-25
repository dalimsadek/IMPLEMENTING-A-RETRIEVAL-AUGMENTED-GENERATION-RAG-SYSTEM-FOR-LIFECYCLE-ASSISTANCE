def build_prompt(chunks, query):
    context = "\n".join(clean_chunk(c) for c in chunks)
    system = (
        "You are a senior Python developer.\n"
        "Only output pure Python code — no explanations.\n"
        "The code must load a trained model, perform inference on structured input data, "
        "and print the results. Do not include model training, data preprocessing, or fitting logic.\n"
        "Avoid markdown, comments, and extra text."
    )

    user = f"#### Context\n{context}\n\n#### Task\n{query}"
    assistant_stub = "import pandas as pd\nimport joblib\n"

    prompt = f"### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n{assistant_stub}"
    return prompt


def conversion_prompt(context, code):
    prompt = f"""You are an embedded systems engineer.
  Convert the following Python inference code into an Arduino C++ sketch that simulates the same logic using the documents.
  Use a TCS34725 or APDS9960 sensor for input, and output results via Serial.
  Assume that the model was trained offline and you must implement the decision logic manually in Arduino C++.

  ### Working Arduino Examples:
  {context}

  ### Python Code:
  {code}

  ### Arduino Sketch:
    """
    return prompt


def correct_code(context: str, code: str, error: str) -> str:
    prompt = f"""You are an expert Arduino developer.

You will receive:
- A buggy Arduino sketch that does **not compile**
- The exact compiler **error message**
- Optionally, working Arduino examples

Your task:
- Find and fix the bug in the sketch based ONLY on the compiler error.
- Do not rewrite logic or change variable names.
- Only return the **corrected sketch** — no explanation or extra text.

### Compiler Error:
{error}

### Buggy Code:
{code}

### Corrected Code:
"""
    return prompt

