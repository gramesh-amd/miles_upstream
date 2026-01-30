"""DeepSeek-V2/V2-Lite weight conversion - reuses V3 converter"""
from .deepseekv3 import convert_deepseekv3_to_hf

# DeepSeek-V2 uses the same weight structure as V3 (minus MTP layers)
convert_deepseekv2_to_hf = convert_deepseekv3_to_hf
