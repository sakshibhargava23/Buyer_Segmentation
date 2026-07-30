from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

SEGMENT_NAMES = {
    0: "Global Investors",
    1: "First-Time Buyers",
    2: "Corporate Buyers",
    3: "Luxury Investors",
}
