from .unidefense import (
    UniDefenseModelEb4,
    UniDefenseModelRes18,
    UniDefenseModelRes50
)
from .fsl import FSL

MODEL = {
    "UDEB4": UniDefenseModelEb4,
    "UDR18": UniDefenseModelRes18,
    "UDR50": UniDefenseModelRes50,
    "FSL": FSL,
}

def load_model(name="UDE"):
    name_upper = name.upper()
    assert name_upper in MODEL, f"Model '{name}' not found."
    print(f"Using model: '{name}'")
    return MODEL[name_upper]
