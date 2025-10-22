#!/usr/bin/env python3
"""
Main pipeline for LLM-based concrete mix design with multi-task support.

This script demonstrates the complete workflow:
1. Load/generate concrete data (compression and tensile)
2. Train traditional ML models
3. Fine-tune an LLM using SFT for multi-task learning
4. Compare all models
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import load_concrete_data
from src.models import train_traditional_models
from src.training import LLMFineTuner
from src.evaluation import compare_all_models, MultiTaskMetrics
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(
        description='LLM Concrete Mix Design Pipeline with Multi-task Support'
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
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--multitask',
        action='store_true',
        default=True,
        help='Enable multi-task learning (compression + tensile)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*70)
    print("LLM Concrete Mix Design Pipeline (Multi-task)")
    print("="*70)
    
    # Step 1: Load data
    print("\n[Step 1/4] Loading concrete data...")
    if args.multitask:
        # Multi-task mode
        dataset = load_concrete_data(args.compression_path, args.tensile_path)
        # For multi-task, we need separate splits for ML models (no NaN) and LLM (with NaN)
        X_train_comp, X_val_comp, X_test_comp, y_train_comp, y_val_comp, y_test_comp = dataset.prepare_single_task_split(
            task='compression',
            train_size=config['data']['train_split'],
            val_size=config['data']['val_split'],
            test_size=config['data']['test_split'],
            random_state=config['data']['random_state']
        )
        X_train_tens, X_val_tens, X_test_tens, y_train_tens, y_val_tens, y_test_tens = dataset.prepare_single_task_split(
            task='tensile',
            train_size=config['data']['train_split'],
            val_size=config['data']['val_split'],
            test_size=config['data']['test_split'],
            random_state=config['data']['random_state']
        )
        # For LLM, we use the full multi-task dataset
        X_train_llm, X_val_llm, X_test_llm, y_train_llm, y_val_llm, y_test_llm = dataset.prepare_train_val_test_split(
            train_size=config['data']['train_split'],
            val_size=config['data']['val_split'],
            test_size=config['data']['test_split'],
            random_state=config['data']['random_state']
        )
    else:
        # Single task mode (backward compatibility)
        dataset = load_concrete_data(args.compression_path)
        X_train, X_test, y_train, y_test = dataset.prepare_train_test_split()
        X_val, y_val = None, None
    
    if args.multitask:
        print(f"  Compression - Training samples: {len(X_train_comp)}")
        print(f"  Compression - Validation samples: {len(X_val_comp)}")
        print(f"  Compression - Test samples: {len(X_test_comp)}")
        print(f"  Tensile - Training samples: {len(X_train_tens)}")
        print(f"  Tensile - Validation samples: {len(X_val_tens)}")
        print(f"  Tensile - Test samples: {len(X_test_tens)}")
        print(f"  LLM - Training samples: {len(X_train_llm)}")
        print(f"  LLM - Validation samples: {len(X_val_llm)}")
        print(f"  LLM - Test samples: {len(X_test_llm)}")
        print(f"  Features: {list(X_train_comp.columns)}")
    else:
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val) if X_val is not None else 0}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {list(X_train.columns)}")
    
    # Step 2: Train traditional ML models
    ml_results = {}
    if not args.skip_ml:
        print("\n[Step 2/4] Training traditional ML models...")
        if args.multitask:
            # For multi-task, we'll train separate models for each task
            print("  Note: Training separate ML models for compression and tensile tasks")
            # Train compression models
            print("  Training compression models...")
            ml_models_comp, ml_results_comp = train_traditional_models(
                X_train_comp, y_train_comp['compressive_strength'], X_test_comp, y_test_comp['compressive_strength']
            )
            # Train tensile models
            print("  Training tensile models...")
            ml_models_tens, ml_results_tens = train_traditional_models(
                X_train_tens, y_train_tens['tensile_strength'], X_test_tens, y_test_tens['tensile_strength']
            )
            # Combine results
            ml_results = {
                'compression': ml_results_comp,
                'tensile': ml_results_tens
            }
            # Save models
            ml_models_comp.save_models(f"{args.output_dir}/ml_models_compression")
            ml_models_tens.save_models(f"{args.output_dir}/ml_models_tensile")
        else:
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
        
        if args.multitask:
            # Prepare multi-task LLM dataset
            llm_data = dataset.prepare_multitask_llm_dataset()
            train_size = int(config['data']['train_split'] * len(llm_data))
            val_size = int(config['data']['val_split'] * len(llm_data))
            
            llm_train_data = llm_data[:train_size]
            llm_val_data = llm_data[train_size:train_size + val_size]
            
            print(f"  LLM training samples: {len(llm_train_data)}")
            print(f"  LLM validation samples: {len(llm_val_data)}")
            
            # Count by task
            train_compression = sum(1 for item in llm_train_data if item.get('task') == 'compression')
            train_tensile = sum(1 for item in llm_train_data if item.get('task') == 'tensile')
            print(f"  Training - Compression: {train_compression}, Tensile: {train_tensile}")
        else:
            # Single task mode
            llm_data = dataset.prepare_llm_dataset()
            train_size = int(0.8 * len(llm_data))
            llm_train_data = llm_data[:train_size]
            llm_val_data = llm_data[train_size:]
            
            print(f"  LLM training samples: {len(llm_train_data)}")
            print(f"  LLM validation samples: {len(llm_val_data)}")
        
        # Initialize and train LLM
        llm_trainer = LLMFineTuner(
            model_name=args.llm_model,
            use_lora=True
        )
        
        train_dataset = llm_trainer.prepare_dataset(llm_train_data)
        val_dataset = llm_trainer.prepare_dataset(llm_val_data)
        
        llm_trainer.train(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            output_dir=f"{args.output_dir}/llm_model",
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Evaluate LLM
        if args.multitask:
            metrics_calculator = MultiTaskMetrics()
            llm_results = metrics_calculator.evaluate_multitask_llm(
                llm_trainer,
                X_test_llm,
                y_test_llm,
                dataset.feature_columns
            )
        else:
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
        
        if args.multitask:
            # Multi-task comparison
            metrics_calculator = MultiTaskMetrics()
            comparison_df = metrics_calculator.generate_metrics_report(
                llm_results['compression'],
                llm_results['tensile'],
                output_dir=args.output_dir
            )
            
            # Create visualizations
            metrics_calculator.plot_multitask_comparison(
                comparison_df,
                save_path=f"{args.output_dir}/multitask_comparison.png"
            )
        else:
            # Single task comparison
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
