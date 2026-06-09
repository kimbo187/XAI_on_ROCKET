"""Run the intrinsic XAI-on-ROCKET experiment from the repository root.

This script is a lightweight entry point for the refactored thesis code.
It executes the without-SHAP baseline on the GunPoint dataset.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    notebook_script = Path(__file__).parent / "notebooks" / "XAI_on_ROCKET_without_SHAP.py"
    runpy.run_path(str(notebook_script), run_name="__main__")
