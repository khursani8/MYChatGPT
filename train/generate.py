from xturing import BaseModel

model = BaseModel.load("lala")

# After this you should be able to run the inference
output = model.generate(texts=["Why are LLM models important"])
print(output)