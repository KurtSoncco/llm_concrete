"""
LLM fine-tuning using Supervised Fine-Tuning (SFT).
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import List, Dict, Optional
import json


class LLMFineTuner:
    """
    Fine-tune an LLM for concrete strength prediction using SFT.
    """
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        device: str = None
    ):
        """
        Initialize the LLM fine-tuner.
        
        Args:
            model_name: HuggingFace model name or path
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            lora_r: LoRA attention dimension
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.use_lora = use_lora
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None
        )
        
        # Apply LoRA if specified
        if use_lora:
            print("Applying LoRA configuration...")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
    
    def prepare_dataset(
        self,
        llm_data: List[Dict[str, str]],
        max_length: int = 512
    ) -> Dataset:
        """
        Prepare dataset for training.
        
        Args:
            llm_data: List of dicts with 'instruction', 'input', 'output', and optionally 'task'
            max_length: Maximum sequence length
            
        Returns:
            HuggingFace Dataset object
        """
        def format_prompt(example):
            """Format example as prompt."""
            prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
            return prompt
        
        # Format prompts
        formatted_data = [
            {'text': format_prompt(example)}
            for example in llm_data
        ]
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize
        def tokenize_function(examples):
            outputs = self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors=None
            )
            outputs['labels'] = outputs['input_ids'].copy()
            return outputs
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=dataset.column_names,
            batched=True
        )
        
        return tokenized_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        output_dir: str = "./models/llm_concrete",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        save_steps: int = 100,
        logging_steps: int = 10,
        eval_dataset: Optional[Dataset] = None
    ):
        """
        Fine-tune the model.
        
        Args:
            train_dataset: Training dataset
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            save_steps: Save checkpoint every N steps
            logging_steps: Log every N steps
            eval_dataset: Optional evaluation dataset
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=2,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            warmup_steps=100,  # Increased warmup
            weight_decay=0.01,
            fp16=self.device == 'cuda',
            push_to_hub=False,
            report_to="none",
            # Additional parameters for better convergence
            gradient_accumulation_steps=2,  # Effective batch size = batch_size * 2
            lr_scheduler_type="cosine",    # Cosine learning rate schedule
            max_grad_norm=1.0,             # Gradient clipping
            dataloader_num_workers=0,       # Avoid multiprocessing issues
            remove_unused_columns=False,   # Keep all columns
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    
    def predict(
        self,
        instruction: str,
        input_text: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7
    ) -> str:
        """
        Generate prediction from the fine-tuned model.
        
        Args:
            instruction: Task instruction
            input_text: Input features
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response part
        response_start = generated_text.find("### Response:")
        if response_start != -1:
            response = generated_text[response_start + len("### Response:"):].strip()
        else:
            response = generated_text
        
        return response
    
    def save_model(self, path: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model and tokenizer."""
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Model loaded from {path}")
