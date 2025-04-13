import os
import numpy as np
import json

workspace = os.path.abspath(__file__)
from site import addsitedir

source_folder = os.path.join("/".join(workspace.split("/")[:-1]), "m4m")
print("source_folder: ", source_folder)
addsitedir(os.path.join(source_folder, "peft"))
import torch
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from trl import SFTTrainer, setup_chat_format


def get_model(use_cache=False):
    base_model = "NousResearch/Meta-Llama-3-8B"

    if torch.cuda.get_device_capability()[0] >= 8:
        attn_implementation = "flash_attention_2"
        torch_dtype = torch.bfloat16
    else:
        attn_implementation = "eager"
        torch_dtype = torch.half

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    print(len(tokenizer), tokenizer.special_tokens_map)
    print("eos", tokenizer.eos_token_id)
    print("bos", tokenizer.bos_token_id)

    pad_token = "<pad>"
    
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|x|>", "<pad>"]})
    print("additional_special_tokens ", tokenizer.special_tokens_map)
    print("eos", tokenizer.eos_token_id)
    print("bos", tokenizer.bos_token_id)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation,
        local_files_only=True
    )

    model.resize_token_embeddings(len(tokenizer))

    pad_id = tokenizer.convert_tokens_to_ids(pad_token)
    tokenizer.pad_token = pad_token
    tokenizer.pad_token_id = pad_id
    print(len(tokenizer))
    print("additional_special_tokens ", tokenizer.special_tokens_map)
    print("eos", tokenizer.eos_token_id)
    print("bos", tokenizer.bos_token_id)

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = use_cache

    model.base_model.get_input_embeddings().weight.requires_grad = False
    np.save("weights/embeded_tokens.npy", model.base_model.get_input_embeddings().weight.cpu().numpy())
    return model, tokenizer, peft_config


def train(model, tokenizer, peft_config):
    from m4m.llama.dataset import MusicDataset

    new_model = "llama-3-8B-m4m-3"

    dataset = MusicDataset(tokenizer,
                           data_path=f"dataset/new_dataset/splited_dataset/train{suffix}.json",
                           feature_folder="dataset/new_dataset/mert_feature")

    training_params = TrainingArguments(
        output_dir=f"./results{suffix}",
        num_train_epochs=6,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        optim="adamw_torch",#paged_adamw_32bit
        save_steps=200,
        logging_steps=5, # 5
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        max_grad_norm=1.,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    trainer = SFTTrainer(
        max_seq_length=2048,
        model=model,
        train_dataset=dataset,
        # eval_dataset=valid_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )
    #last_checkpoint = get_last_checkpoint("./results-4")

    trainer.train()
    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)

    return new_model

if __name__ == "__main__":
    model, tokenizer, peft_config = get_model()
    suffix = "_3000_sample"
    new_model = train(model, tokenizer, peft_config)
