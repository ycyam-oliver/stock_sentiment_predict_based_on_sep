from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training # prepare_model_for_int8_training
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline, BitsAndBytesConfig
from transformers import LlamaTokenizer, LlamaConfig, LlamaForSequenceClassification, LlamaForCausalLM

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
import os


# DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "</s>"
# DEFAULT_UNK_TOKEN = "</s>"

tqdm.pandas()


def tuning_lm_with_rl(args):
    script_args = args

    reward_model_name = script_args.reward_model_name

    # dataset_name = "lvwerra/stack-exchange-paired"
    dataset_name = script_args.datasets_dir
    print("dataset_name: ", dataset_name)

    config = PPOConfig(
        model_name=script_args.rl_base_model,
        learning_rate=script_args.rl_learning_rate,
        log_with=script_args.log_with,
        batch_size=script_args.batch_size,
        mini_batch_size=script_args.mini_batch_size,
        gradient_accumulation_steps=script_args.rl_gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=script_args.early_stopping,
        target_kl=script_args.target_kl,
        ppo_epochs=script_args.ppo_epochs,
        seed=script_args.seed,
    )

    # train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train")
    # train_dataset = train_dataset.select(range(100000))
    train_dataset = load_dataset(dataset_name, split="train")
    # train_dataset = train_dataset.select(range(100))
    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    # sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16, "truncation": True}
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1, "truncation": True}

    # tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
    # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
    # only for this model.

    if "llama" in script_args.tokenizer_name or "vicuna" in script_args.rl_base_model or "Vicuna" in script_args.rl_base_model:
        tokenizer = LlamaTokenizer.from_pretrained(script_args.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)


    # if "llama" in script_args.tokenizer_name or "vicuna" in script_args.rl_base_model or "Vicuna" in script_args.rl_base_model:
    #     tokenizer.add_special_tokens(
    #         {
    #             "eos_token": DEFAULT_EOS_TOKEN,
    #             "bos_token": DEFAULT_BOS_TOKEN,
    #             "unk_token": DEFAULT_UNK_TOKEN,
    #             "pad_token": DEFAULT_PAD_TOKEN,
    #         }
    #     )
    # else:
    #     tokenizer.pad_token = tokenizer.eos_token


    # Below is an example function to build the dataset. In our case, we use the IMDB dataset
    # from the `datasets` library. One should customize this function to train the model on
    # its own dataset.
    def build_dataset(
        tokenizer, dataset_name="lvwerra/stack-exchange-paired", input_min_text_length=2, input_max_text_length=8
    ):
        """
        Build dataset for training. This builds the dataset from `load_dataset`, one should
        customize this function to train the model on its own dataset.

        Args:
            dataset_name (`str`):
                The name of the dataset to be loaded.

        Returns:
            dataloader (`torch.utils.data.DataLoader`):
                The dataloader for the dataset.
        """

        # load imdb with datasets
        # ds = load_dataset(dataset_name, data_dir="data/rl", split="train")
        ds = load_dataset(dataset_name, split="train")
        original_columns = ds.column_names
        num_proc = 1 #24

        def preprocess_function(examples):
            new_examples = {
                "query": [],
                "input_ids": [],
            }
            # for question in examples["question"]:
            for question in examples["user_input"]:
                query = "Question: " + question + "\n\nAnswer: "
                tokenized_question = tokenizer(query, truncation=True)
                new_examples["query"].append(query)
                new_examples["input_ids"].append(tokenized_question["input_ids"])

            return new_examples

        ds = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns,
        )
        # ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

        ds.set_format(type="torch")
        return ds


    # We retrieve the dataloader by calling the `build_dataset` function.
    # dataset = build_dataset(tokenizer)
    dataset = build_dataset(tokenizer, dataset_name=dataset_name)


    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])


    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    current_device = Accelerator().local_process_index

    lora_config = LoraConfig(
        r=8, #16,
        lora_alpha=16, #32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        load_in_4bit=True, # Oli
        # device_map={"": current_device},
        device_map="auto",
        peft_config=lora_config,
        # layer_norm_names=[],
        # torch_dtype=torch.float16, # Oli
        quantization_config=BitsAndBytesConfig(
            # load_in_4bit=True,  # Oli
            # bnb_4bit_quant_type="nf4", # Oli
            # bnb_4bit_use_double_quant=True, # Oli
            # bnb_4bit_compute_dtype=torch.bfloat16, # Oli
            # bnb_4bit_quant_storage=torch.bfloat16, # Oli
            llm_int8_enable_fp32_cpu_offload=True) 
    )
    # model = AutoModelForCausalLMWithValueHead.from_pretrained(
    # config.model_name,
    # load_in_4bit=True,
    # # device_map={"": current_device},
    # device_map="auto",
    # peft_config=lora_config,
    # # layer_norm_names=[],
    # # torch_dtype=torch.float16,
    # quantization_config=BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True))
    print("finetune model: ", config.model_name, type(model))
    print("finetune model's is_loaded_in_4bit: ", model.is_loaded_in_4bit)

    optimizer = None
    if script_args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
        )
    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    # We then build the sentiment analysis pipeline, passing the model name and the
    # sentiment analysis pipeline arguments. Let's also make sure to set the device
    # to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
    print("device: ", device)


    print("reward_model_name: ", reward_model_name)
    #! my self code to try peft reward model
    # reward_model = LlamaForSequenceClassification.from_pretrained(
    #     reward_model_name,
    #     load_in_4bit=True,
    #     device_map="auto",
    #     torch_dtype=torch.float16,
    # )
    # print("reward_model: ", type(reward_model))
    # print("reward_model is_loaded_in_4bit: ", reward_model.is_loaded_in_4bit)

    # reward_model = prepare_model_for_int8_training(reward_model)
    # reward_model_config = LlamaConfig.from_pretrained(reward_model_name)

    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=reward_model_name,
        # model=reward_model,
        # config=reward_model_config,
        device_map="auto",
        # TypeError: LlamaForSequenceClassification.__init__() got an unexpected keyword argument 'peft_config'
        model_kwargs={"load_in_4bit": True},
        tokenizer=tokenizer,
        # torch_dtype=torch.float16,
    )

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        # "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": 100_000,
    }
    output_min_length = 32
    output_max_length = script_args.output_max_length
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        question_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]

        # Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")

        ppo_trainer.save_pretrained(script_args.output_dir + "step_saved")
