#!/usr/bin/env python3
"""
Multi-task LLM fine-tuning for concrete strength prediction.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import load_concrete_data
from src.training import LLMFineTuner
from src.evaluation import MultiTaskMetrics
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune LLM for multi-task concrete strength prediction'
    )
    parser.add_argument(
        '--compression-path',
        type=str,
        default='data/Data_Compresion_Concreto.csv',
        help='Path to compression CSV file'
    )
    parser.add_argument(
        '--tensile-path',
        type=str,
        default='data/Data_Traccion_Concreto.csv',
        help='Path to tensile CSV file'
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
        default='./models/multitask_llm',
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
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*70)
    print("Multi-task LLM Fine-tuning for Concrete Strength Prediction")
    print("="*70)
    
    # Load multi-task data
    print("\n[Step 1/5] Loading multi-task concrete data...")
    dataset = load_concrete_data(args.compression_path, args.tensile_path)
    
    print(f"Dataset shape: {dataset.data.shape}")
    print(f"Compression samples: {len(dataset.compression_data) if dataset.compression_data is not None else 0}")
    print(f"Tensile samples: {len(dataset.tensile_data) if dataset.tensile_data is not None else 0}")
    
    # Prepare train/val/test split
    print("\n[Step 2/5] Preparing train/validation/test split...")
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.prepare_train_val_test_split(
        train_size=config['data']['train_split'],
        val_size=config['data']['val_split'],
        test_size=config['data']['test_split'],
        random_state=config['data']['random_state']
    )
    
    # Prepare multi-task LLM dataset
    print("\n[Step 3/5] Preparing multi-task LLM dataset...")
    llm_data = dataset.prepare_multitask_llm_dataset()
    
    # Split LLM data for training and evaluation
    train_size = int(config['data']['train_split'] * len(llm_data))
    val_size = int(config['data']['val_split'] * len(llm_data))
    
    llm_train_data = llm_data[:train_size]
    llm_val_data = llm_data[train_size:train_size + val_size]
    llm_test_data = llm_data[train_size + val_size:]
    
    print(f"LLM training samples: {len(llm_train_data)}")
    print(f"LLM validation samples: {len(llm_val_data)}")
    print(f"LLM test samples: {len(llm_test_data)}")
    
    # Count samples by task
    train_compression = sum(1 for item in llm_train_data if item.get('task') == 'compression')
    train_tensile = sum(1 for item in llm_train_data if item.get('task') == 'tensile')
    print(f"Training - Compression: {train_compression}, Tensile: {train_tensile}")
    
    # Initialize LLM trainer
    print(f"\n[Step 4/5] Initializing LLM: {args.model_name}")
    llm_trainer = LLMFineTuner(
        model_name=args.model_name,
        use_lora=args.use_lora
    )
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = llm_trainer.prepare_dataset(llm_train_data)
    val_dataset = llm_trainer.prepare_dataset(llm_val_data)
    
    # Train
    print("\nStarting multi-task fine-tuning...")
    llm_trainer.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate on test set
    print("\n[Step 5/5] Evaluating multi-task model...")
    metrics_calculator = MultiTaskMetrics()
    
    llm_results = metrics_calculator.evaluate_multitask_llm(
        llm_trainer,
        X_test,
        y_test,
        dataset.feature_columns
    )
    
    # Generate comprehensive report
    comparison_df = metrics_calculator.generate_metrics_report(
        llm_results['compression'],
        llm_results['tensile'],
        output_dir='./results'
    )
    
    # Create visualizations
    metrics_calculator.plot_multitask_comparison(
        comparison_df,
        save_path='./results/multitask_comparison.png'
    )
    
    print("\n" + "="*70)
    print("Multi-task LLM training completed successfully!")
    print("="*70)
    print(f"Model saved to: {args.output_dir}")
    print(f"Results saved to: ./results/")


if __name__ == '__main__':
    main()
