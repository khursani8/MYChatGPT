
from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import BaseModel
from xturing.models.causal import CausalModel
from xturing.engines.causal import CausalEngine
from typing import Optional, Union
from pathlib import Path

models = [
    "mesolitica/finetune-paraphrase-t5-tiny-standard-bahasa-cased",
    "mesolitica/finetune-paraphrase-t5-small-standard-bahasa-cased",
    "mesolitica/finetune-paraphrase-t5-base-standard-bahasa-cased",

    "mesolitica/finetune-isi-penting-generator-t5-small-standard-bahasa-cased",
    "mesolitica/finetune-isi-penting-generator-t5-base-standard-bahasa-cased",
]

# Initializes the model
class MyEngine(CausalEngine):
    config_name: str = "gpt2_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(model_name=models[0], weights_path=weights_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token

class MyModel(CausalModel):
    config_name: str = models[0]

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MyEngine.config_name, weights_path)

model = MyModel()

instruction_dataset = InstructionDataset("alpaca_data")
finetuning_config = model.finetuning_config()
finetuning_config.batch_size = 8
finetuning_config.num_train_epochs = 5
finetuning_config.learning_rate = 4e-5
finetuning_config.max_length = 512
model.finetune(dataset=instruction_dataset)

# Perform inference
output = model.generate(texts=["Why LLM models are becoming so important?"])
print("Why LLM models are becoming so important?")
print("Generated output by the model: {}".format(output))

# Perform inference
output = model.generate(texts=["Beri cadangan untuk menangani rasa bosan semasa Perintah Kawalan Pergerakan (PKP)."])
print("Beri cadangan untuk menangani rasa bosan semasa Perintah Kawalan Pergerakan (PKP).")
print("Generated output by the model: {}".format(output))