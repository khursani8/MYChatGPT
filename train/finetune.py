
from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import BaseModel
from xturing.models.llama import LlamaLoraInt8,LlamaInt8,Llama
from xturing.models.gpt2 import GPT2LoraInt8


# Initializes the model
# model = BaseModel.create("llama_lora_int8")
# model = Llama()
# model = LlamaInt8()
model = LlamaLoraInt8()
# model = GPT2LoraInt8()

# https://xturing.stochastic.ai/finetune/configure/
finetuning_config = model.finetuning_config()
finetuning_config.batch_size = 4
finetuning_config.num_train_epochs = 20
finetuning_config.learning_rate = 4e-5
finetuning_config.weight_decay = 1e-5
finetuning_config.max_length = 1024
# finetuning_config.optimizer_name = "adamw"
# finetuning_config.output_dir = "training_dir/"
model.finetuning_args = finetuning_config

# Perform inference
print("-"*40)
output = model.generate(texts=["Why LLM models are becoming so important?"])[0]
print("Why LLM models are becoming so important?")
print("Generated output by the model: {}".format(output))
print("-"*40)

instruction_dataset = InstructionDataset("alpaca_data")
model.finetune(dataset=instruction_dataset)

model.save("lala")

# Perform inference
output = model.generate(texts=["Beri cadangan untuk menangani rasa bosan semasa Perintah Kawalan Pergerakan (PKP)."])
print("Beri cadangan untuk menangani rasa bosan semasa Perintah Kawalan Pergerakan (PKP).")
print("Generated output by the model: {}".format(output))