#!/usr/bin/env python3
"""
Enhanced Multi-task LLM Training Script with Synthetic Data and Better Prompts

This script demonstrates:
- Synthetic data generation for increased training samples
- Few-shot prompt engineering
- Support for larger models
- Enhanced data augmentation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import load_concrete_data
from src.training import LLMFineTuner
from src.evaluation import MultiTaskMetrics
import argparse
import yaml
import torch

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced multi-task LLM training with synthetic data and better prompts'
    )
    parser.add_argument('--compression-path', type=str, default='data/Data_Compresion_Concreto.csv', 
                       help='Path to compression CSV file')
    parser.add_argument('--tensile-path', type=str, default='data/Data_Traccion_Concreto.csv', 
                       help='Path to tensile CSV file')
    parser.add_argument('--model-name', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', 
                       help='HuggingFace model name')
    parser.add_argument('--output-dir', type=str, default='./models/enhanced_multitask_llm', 
                       help='Output directory for fine-tuned model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    parser.add_argument('--use-lora', action='store_true', default=True, help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--synthetic-samples', type=int, default=2000, 
                       help='Number of synthetic samples to generate')
    parser.add_argument('--eval-samples', type=int, default=50, 
                       help='Number of samples for evaluation (for speed)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--skip-ml', action='store_true', help='Skip traditional ML training')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Enhanced Multi-task LLM Training with Synthetic Data")
    print("="*80)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: Running on CPU will be very slow. Consider using GPU.")
        if "7b" in args.model_name.lower() or "7B" in args.model_name:
            print("Error: 7B models require GPU. Please use a smaller model or GPU.")
            return
    
    # Step 1: Load data with synthetic augmentation
    print("\n[Step 1/4] Loading and augmenting data...")
    dataset = load_concrete_data(args.compression_path, args.tensile_path)
    
    # Use synthetic data augmentation
    dataset.load_data_with_synthetic(args.synthetic_samples)
    
    # Prepare splits
    X_train_llm, X_val_llm, X_test_llm, y_train_llm, y_val_llm, y_test_llm = dataset.prepare_train_val_test_split(
        train_size=config['data']['train_split'],
        val_size=config['data']['val_split'],
        test_size=config['data']['test_split'],
        random_state=config['data']['random_state']
    )
    
    print(f"Final dataset: {len(X_train_llm)} train, {len(X_val_llm)} val, {len(X_test_llm)} test")
    
    # Step 2: Prepare LLM dataset with enhanced prompts
    print("\n[Step 2/4] Preparing enhanced LLM dataset...")
    llm_data = dataset.prepare_multitask_llm_dataset(augment_data=True)
    
    # Split LLM data
    train_size = int(config['data']['train_split'] * len(llm_data))
    val_size = int(config['data']['val_split'] * len(llm_data))
    
    llm_train_data = llm_data[:train_size]
    llm_val_data = llm_data[train_size:train_size + val_size]
    
    print(f"LLM training samples: {len(llm_train_data)}")
    print(f"LLM validation samples: {len(llm_val_data)}")
    
    # Count by task
    train_compression = sum(1 for item in llm_train_data if item.get('task') == 'compression')
    train_tensile = sum(1 for item in llm_train_data if item.get('task') == 'tensile')
    print(f"Training - Compression: {train_compression}, Tensile: {train_tensile}")
    
    # Step 3: Train LLM
    print("\n[Step 3/4] Training enhanced LLM...")
    print(f"Model: {args.model_name}")
    print(f"LoRA: {args.use_lora}")
    
    llm_trainer = LLMFineTuner(
        model_name=args.model_name,
        use_lora=args.use_lora
    )
    
    train_dataset = llm_trainer.prepare_dataset(llm_train_data)
    val_dataset = llm_trainer.prepare_dataset(llm_val_data)
    
    print("Starting enhanced training...")
    llm_trainer.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=config['llm']['training']['learning_rate']
    )
    
    # Step 4: Evaluate
    print("\n[Step 4/4] Evaluating enhanced LLM...")
    metrics_calculator = MultiTaskMetrics()
    llm_results = metrics_calculator.evaluate_multitask_llm_fast(
        llm_trainer,
        X_test_llm,
        y_test_llm,
        dataset.feature_columns,
        max_samples=args.eval_samples  # Configurable evaluation samples
    )
    
    print("\nEnhanced Multi-task LLM Results:")
    print("="*50)
    print("Compression Metrics:", llm_results['compression'])
    print("Tensile Metrics:", llm_results['tensile'])
    
    # Generate reports
    comparison_df = metrics_calculator.generate_metrics_report(
        llm_results['compression'],
        llm_results['tensile'],
        output_dir=args.output_dir
    )
    
    metrics_calculator.plot_multitask_comparison(
        comparison_df,
        save_path=f"{args.output_dir}/enhanced_multitask_comparison.png"
    )
    
    print(f"\nResults saved to: {args.output_dir}")
    print("Enhanced training completed!")
    
    # Show improvement suggestions
    print("\n" + "="*80)
    print("IMPROVEMENT SUGGESTIONS:")
    print("="*80)
    
    compression_r2 = llm_results['compression'].get('r2', -999)
    tensile_r2 = llm_results['tensile'].get('r2', -999)
    
    if compression_r2 < 0.5:
        print("🔧 Compression R² is low. Consider:")
        print("   - Using a larger model (7B parameters)")
        print("   - More training epochs (5-10)")
        print("   - Increasing synthetic data (5000+ samples)")
        print("   - Fine-tuning learning rate")
    
    if tensile_r2 < 0.5:
        print("🔧 Tensile R² is low. Consider:")
        print("   - Using a larger model (7B parameters)")
        print("   - More training epochs (5-10)")
        print("   - Increasing synthetic data (5000+ samples)")
        print("   - Fine-tuning learning rate")
    
    if compression_r2 > 0.5 and tensile_r2 > 0.5:
        print("🎉 Great results! Consider:")
        print("   - Testing on more diverse data")
        print("   - Deploying for production use")
        print("   - Adding more concrete types")

if __name__ == '__main__':
    main()
