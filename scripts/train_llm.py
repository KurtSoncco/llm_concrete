#!/usr/bin/env python3
"""
Fine-tune LLM for concrete strength prediction.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import load_concrete_data
from src.training import LLMFineTuner
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune LLM for concrete strength prediction'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to concrete dataset CSV'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./models/llm_model',
        help='Output directory for fine-tuned model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size'
    )
    parser.add_argument(
        '--use-lora',
        action='store_true',
        default=True,
        help='Use LoRA for efficient fine-tuning'
    )
    
    args = parser.parse_args()
    
    print("Loading data...")
    dataset = load_concrete_data(args.data_path)
    llm_data = dataset.prepare_llm_dataset()
    
    # Split for training and evaluation
    train_size = int(0.8 * len(llm_data))
    llm_train_data = llm_data[:train_size]
    llm_eval_data = llm_data[train_size:]
    
    print(f"Training samples: {len(llm_train_data)}")
    print(f"Evaluation samples: {len(llm_eval_data)}")
    
    print(f"\nInitializing LLM: {args.model_name}")
    llm_trainer = LLMFineTuner(
        model_name=args.model_name,
        use_lora=args.use_lora
    )
    
    print("\nPreparing datasets...")
    train_dataset = llm_trainer.prepare_dataset(llm_train_data)
    eval_dataset = llm_trainer.prepare_dataset(llm_eval_data)
    
    print("\nStarting fine-tuning...")
    llm_trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
