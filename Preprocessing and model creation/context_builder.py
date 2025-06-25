import pandas as pd
import json
from typing import Dict, List


class ContextBuilder:
    def __init__(
        self,
        csv_path: str,
        task_description: str,
        feature_description: Dict[str, str],
        label_description: str,
        notes: str = "",
        num_preview_rows: int = 5,
    ):
        self.csv_path = csv_path
        self.task_description = task_description
        self.feature_description = feature_description
        self.label_description = label_description
        self.notes = notes
        self.num_preview_rows = num_preview_rows

        self.df = pd.read_csv(csv_path)
        self.context = self._build_context()

    def _build_context(self) -> Dict:
        preview = self.df.head(self.num_preview_rows).to_dict(orient="records")

        return {
            "task": self.task_description,
            "features": list(self.feature_description.keys()),
            "feature_explanation": self.feature_description,
            "label_explanation": self.label_description,
            "sample_data": preview,
            "notes": self.notes,
            "source_file": self.csv_path,
        }

    def get_context(self) -> Dict:
        return self.context

    def get_context_as_text(self) -> str:
        context = self.get_context()
        lines = [
            f"Task: {context['task']}",
            "Features:",
        ] + [
            f"  - {key}: {desc}"
            for key, desc in context["feature_explanation"].items()
        ] + [
            f"Label Info: {context['label_explanation']}",
            f"Notes: {context['notes']}",
            "Sample Data:"
        ] + [
            f"  - {row}" for row in context["sample_data"]
        ]
        return "\n".join(lines)

    def save_to_json(self, output_path: str = "context.json"):
        with open(output_path, "w") as f:
            json.dump(self.context, f, indent=2)
