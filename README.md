# MYChatGPT

## Translated data for training chatgpt like model

During March I decide to learn NLP because I got some task related to NLP. 
In this repo is the data I collected for trying to understand how to develop chatgpt like model and my notes.

I did not share the model because I tried most of the repo that claim able to be as good like chatgpt is just a model that overfit to chatgpt response dataset which is more than 50k rows. From what I can see they did not even do RLHF for their model, only finetune whole model or train on LoRa model.

You can try train llama/alpaca model with this data and share the model link here after hosting in huggingface.

NLP is not my strong suit, let me know if I'm doing something wrong by creating an issue or make a pull request.

# Data

1. Stanford alpaca like data https://github.com/tatsu-lab/stanford_alpaca/

- [Malay Data](./stanford/malay.json) # 1750

- [Malay Seed Task](./stanford/seed_task_ms.jsonl) # 416

2. gpt4all https://github.com/nomic-ai/gpt4all

- [Translated malay](./gpt4all/data_ms.jsonl) # 1940

- [Original english](./gpt4all/data.jsonl) # 50k+

3. kochatgpt https://github.com/airobotlab/KoChatGPT

- [Supervised dataset](./kochatgpt/mschatgpt_1_SFT.jsonl) # 2350

- [Supervised conversation dataset](./kochatgpt/mychatgpt_1_SFT_conversation.jsonl) # 442

- [Reward model dataset](./kochatgpt/mschatgpt_2_RM.jsonl) # 397

- [PPO dataset](./kochatgpt/mschatgpt_3_PPO.jsonl) # 269

4. chatdoctor https://github.com/Kent0n-Li/ChatDoctor/

- [Translated malay](./chatdoctor/chatdoctor5k_ms.json) # 908

- [Original english](./chatdoctor/chatdoctor5k.json) # 5460

Usage and License Notices: MYChatGPT is intended and licensed for research use only. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.

## Training

So far I only tried:

1. https://github.com/johnsmith0031/alpaca_lora_4bit
- This repo easy to setup and train model, also got nice gui for different text generation like chatbot, instruction etc
- Can play around with lot of parameters like beam search parameter, len penalty etc

1. https://github.com/stochasticai/xturing/
- Easy to train
- Until now cannot make the model generate something sentence

## Notes

Why chatgpt different with other text generation model?
- OpenAI finetune their gpt model with reinforcement learning.
- At first they train using normal supervised language model to predict next sentence.
- At 2nd stage they train a model called critic or if someone not familiar with RL can also think of it like discriminator for GAN
- This critic will act like a human rater telling the model which response is good and bad.
- From this response we will finetune using RL technique call PPO which is commonly used to optimize non differentiable function
- I at some point also use PPO before to improve my speech to text decoder based on REINFORCE paper, the problem with my method is that I'm sampling from the topk beam search instead of sampling from human annotator rank.
- Now you got a model that able to tell if the response is good or not, you don't need to hire thousand of people to rank your billions of response dataset anymore.
- The 3rd stage is to finetune your typical Language model based on the reward given by your 2nd stage model. From here on we don't need human anymore.
- At the same time will also include kldiv loss between your 1st stage language model which is frozen and 3rd stage model we currently finetuning so that the model will be regularize as to not output garbage response during finetuning stage.

1st stage: Supervised Fine Tuning

2nd stage: Reward model

3rd stage: PPO learning

![alt text](https://cdn.openai.com/instruction-following/draft-20220126f/methods-mobile.svg "Title")

# TODO

[] Setup argilla annotation site https://docs.argilla.io/en/latest/tutorials/notebooks/training-textgeneration-unstructured.html (easy mlops setup)

[] Write a script to ask chatgpt whether an output is correct for the given instruction and input

[] Train model using trlx https://wandb.ai/carperai/summarize_RLHF/reports/Implementing-RLHF-Learning-to-Summarize-with-trlX--VmlldzozMzAwODM2

[] Train model using colossal-ai https://www.hpc-ai.tech/blog/colossal-ai-chatgpt

[] Train using xturing and finetune malaya model using RLHF

[] Train using xturing and finetune gpt4all model using RLHF

[] Clean dataset

Today already 1st april I decide to study Speech domain, I will work on above list next June or July, if you guys want to help contribute feel free to do so.
I would say during June and July there might be someone or company like huggingface make it easier to train model using this kind of dataset.

I will update this repo from time to time if I did something related that can be share here.

# In case you want to make the data better

## What I can contribute:
1. Add more seed task in stanford/seed_task_ms.jsonl
2. Fix translated malay word from indonesian to malay in stanford/malay.json, gpt4all/data_ms.jsonl, kochatgpt/*.jsonl
3. Fix logic error or math error
4. Since this is translated the thing related to West information and Korean need to be remove or change to Malaysia related information
5. Remove row containing unicode or rewrite it until it make sense.

## What other possible issue with the dataset? https://github.com/gururise/AlpacaDataCleaned#issues-with-the-original-dataset
1. Hallucinations: Many instructions in the original dataset had instructions referencing data on the internet, which just caused GPT3 to hallucinate an answer.
2. Empty outputs: Some entries in the original dataset had empty outputs.
3. Empty code examples: Some descriptions in the original dataset were missing code examples, making it difficult to understand the intended behavior of the code.
4. Instructions to generate something it not capable of like generate image: Some descriptions in the original dataset included instructions to generate images, something obviously not possible.
5. N/A outputs: Some code snippets in the original dataset had N/A outputs.
6. Inconsistent input field: The original dataset had inconsistent usage of the input field when it was supposed to be empty.
```
"input":"<no input>"
"input":"No input"
"input":"noinput"
"input":"<noinput>"
```
7. Wrong answers: Some instructions/questions in the original dataset had incorrect answers. About 80% of the math problems are estimated to have incorrect answers.
8. Non-Sensical/Unclear instructions: Many instructions are unclear, we try to clarify (or re-write) if instructions are non-sensical. Instructions that are slightly unclear, but where one could deduce the meaning are not altered.
9. Extraneous escape and control characters: The original dataset had several entries with extraneous escape and control characters.

## How to edit the file?
press command in this repo page, if you read this text from github you already in this repo page, just hit '.' on your keyboard
or
go to here github.dev

## What is seed task?
You can think of it list of instruction we want to train LLM model to reply
From that seed task we can do augmentation by asking openai model to create similar task and that is malay.json
We finetune malay.json to LLM to answer any malay related question.