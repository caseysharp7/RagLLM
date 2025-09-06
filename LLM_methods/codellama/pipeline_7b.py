# attempt at running the model at full precision split between two gpus, the work is still in progress
import torch
import torch.distributed
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.pipelining import SplitPoint, pipeline, ScheduleGPipe
print("working 1")
model_name = "codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("working 2")
tokenizer.pad_token = tokenizer.eos_token
print("working 3")
prompts = ("try this ", "nah ion think so...**&",)
print("working 4")
rank = 0
world_size = 2
print("working 5")
dev = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
torch.distributed.init_process_group(rank=rank, world_size = world_size)
print("working 6")
#model.to(dev).eval()
