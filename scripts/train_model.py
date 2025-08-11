from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

def train_model(model, tokenizer, dataset, max_seq_length=2048):
    """
    Train the model using SFTTrainer.
    """
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
        ),
    )
    return trainer.train()

def save_model(model, tokenizer, output_path="ggufmodel"):
    """
    Save the trained model in GGUF format.
    """
    model.save_pretrained_gguf(output_path, tokenizer, quantization_method="q4_k_m")
