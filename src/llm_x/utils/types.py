from typing import Dict

DTYPE_BYTE_MAP: Dict[str, int] = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "Q8_0": 1, 
}

def get_bytes_per_element(dtype_str: str) -> int:
    return DTYPE_BYTE_MAP.get(dtype_str.upper(), 2)