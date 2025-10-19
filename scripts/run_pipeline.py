#!/usr/bin/env python3
"""
Main pipeline for LLM-based concrete mix design.

This script demonstrates the complete workflow:
1. Load/generate concrete data
2. Train traditional ML models
3. Fine-tune an LLM using SFT
4. Compare all models
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import load_concrete_data
from src.models import train_traditional_models
from src.training import LLMFineTuner
from src.evaluation import compare_all_models
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='LLM Concrete Mix Design Pipeline'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to concrete dataset CSV (if None, uses synthetic data)'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        help='HuggingFace model name for LLM'
    )
    parser.add_argument(
        '--skip-ml',
        action='store_true',
        help='Skip training traditional ML models'
    )
    parser.add_argument(
        '--skip-llm',
        action='store_true',
        help='Skip training LLM model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs for LLM'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for LLM training'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("LLM Concrete Mix Design Pipeline")
    print("="*70)
    
    # Step 1: Load data
    print("\n[Step 1/4] Loading concrete data...")
    dataset = load_concrete_data(args.data_path)
    X_train, X_test, y_train, y_test = dataset.prepare_train_test_split()
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {list(X_train.columns)}")
    
    # Step 2: Train traditional ML models
    ml_results = {}
    if not args.skip_ml:
        print("\n[Step 2/4] Training traditional ML models...")
        ml_models, ml_results = train_traditional_models(
            X_train, y_train, X_test, y_test
        )
        # Save models
        ml_models.save_models(f"{args.output_dir}/ml_models")
    else:
        print("\n[Step 2/4] Skipping traditional ML models...")
    
    # Step 3: Train LLM with SFT
    llm_results = {}
    if not args.skip_llm:
        print("\n[Step 3/4] Fine-tuning LLM with SFT...")
        
        # Prepare LLM dataset
        llm_data = dataset.prepare_llm_dataset()
        train_size = int(0.8 * len(llm_data))
        llm_train_data = llm_data[:train_size]
        llm_eval_data = llm_data[train_size:]
        
        print(f"  LLM training samples: {len(llm_train_data)}")
        print(f"  LLM eval samples: {len(llm_eval_data)}")
        
        # Initialize and train LLM
        llm_trainer = LLMFineTuner(
            model_name=args.llm_model,
            use_lora=True
        )
        
        train_dataset = llm_trainer.prepare_dataset(llm_train_data)
        eval_dataset = llm_trainer.prepare_dataset(llm_eval_data)
        
        llm_trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=f"{args.output_dir}/llm_model",
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Evaluate LLM
        from src.evaluation import ModelComparison
        comparator = ModelComparison()
        llm_results = comparator.evaluate_llm(
            llm_trainer,
            X_test,
            y_test,
            dataset.feature_columns
        )
    else:
        print("\n[Step 3/4] Skipping LLM training...")
    
    # Step 4: Compare models
    if ml_results and llm_results:
        print("\n[Step 4/4] Comparing all models...")
        comparison_df = compare_all_models(
            ml_results,
            llm_results,
            y_test.values,
            output_dir=args.output_dir
        )
        
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(comparison_df.to_string(index=False))
        print(f"\nResults saved to: {args.output_dir}/")
    else:
        print("\n[Step 4/4] Skipping comparison (need both ML and LLM results)...")
    
    print("\n" + "="*70)
    print("Pipeline completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
