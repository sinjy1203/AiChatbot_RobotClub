import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description="Quantize AWQ model")

parser.add_argument("--model", required=True)
parser.add_argument("--save-path", required=True)

args = parser.parse_args()

model_path = args.model
quant_path = args.save_path
# model_path = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
# quant_path = "EEVE-Korean-Instruct-10.8B-v1.0-quantized"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True})
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
