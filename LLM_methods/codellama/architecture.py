from transformers import AutoModelForCausalLM

model_name = "codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)

print(model)
