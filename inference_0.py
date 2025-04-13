import os
import numpy as np

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


def print_text(x):
    token = "<|x|>"
    db_token = token + token
    while len(x.split(db_token)) > 1:
        x = str.replace(x, db_token, token)
    x = str.replace(x, token, "*")
    print(x)
    return x


def format_input(data, device):
    return torch.from_numpy(np.array(data))[None, ...].to(device)


def get_model(use_cache=False):
    base_model = "NousResearch/Meta-Llama-3-8B"

    if torch.cuda.get_device_capability()[0] >= 8:
        attn_implementation = "flash_attention_2"
        torch_dtype = torch.bfloat16
    else:
        attn_implementation = "eager"
        torch_dtype = torch.float16

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
    print("<|end_of_text|>", tokenizer.convert_tokens_to_ids("<|end_of_text|>"))
    print("<|eot_id|>", tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    print("eos", tokenizer.eos_token_id)
    print("bos", tokenizer.bos_token_id)

    pad_token = "<|eot_id|>"

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

    print(len(tokenizer))
    print("additional_special_tokens ", tokenizer.special_tokens_map)
    print("eos", tokenizer.eos_token_id)
    print("bos", tokenizer.bos_token_id)

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = use_cache

    model.base_model.get_input_embeddings().weight.requires_grad = False
    # np.save("weights/embeded_tokens.npy", model.base_model.get_input_embeddings().weight.cpu().numpy())
    return model, tokenizer, peft_config


def inference(model, tokenizer, peft_config, output_path):
    from m4m.llama.dataset import MusicDataset

    base_model = model
    if torch.cuda.get_device_capability()[0] >= 8:
        attn_implementation = "flash_attention_2"
        torch_dtype = torch.bfloat16
    else:
        attn_implementation = "eager"
        torch_dtype = torch.float16

    checkpoint_dir = "results_3000_sample/checkpoint-600"
    model = PeftModel.from_pretrained(
        model=base_model,
        model_id=checkpoint_dir,
        peft_config=peft_config,

    )
    model.load_music_encoder(checkpoint_dir)
    dataset = MusicDataset(tokenizer,
                           inference=True,
                           data_path="dataset/new_dataset/splited_dataset/test_3000_sample.json", # test_gianttempo  
                           feature_folder="dataset/new_dataset/encodec_feature",
                           )

    n_samples = 0
    outputs = []
    for data in dataset.inference():
        print("+++++++++++++++++++++++++++++++++++++++++++++++++")
        outputs.append(print_text(f"[filename  ]: {data['filename']}"))
        outputs.append(print_text(f"[Question  ]: {data['Q']}"))
        outputs.append(print_text(f"[Answer Ref]: {data['A']}"))

        input_ids = format_input(data["input_ids"], model.music_encoder.device)
        clap_rep = format_input(data["clap_rep"], model.music_encoder.device)
        pos_id = format_input(data["pos_id"], model.music_encoder.device)
        with torch.no_grad():
            output = model.generate(input_ids=input_ids,
                                    clap_rep=clap_rep,
                                    pos_id=pos_id,
                                    temperature=0.0,
                                    max_length=2048)

            text = tokenizer.decode(list(output[0].cpu().numpy()))
        outputs.append(print_text(f"[Answer Est]: {text}"))
        n_samples += 1
        with open(output_path, "w") as f:
            f.write("\n".join(outputs))


if __name__ == "__main__":
    model, tokenizer, peft_config = get_model()
    output_path = "dataset/results/test_3000"
    inference(model, tokenizer, peft_config, output_path)
