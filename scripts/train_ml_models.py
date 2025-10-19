#!/usr/bin/env python3
"""
Train only traditional ML models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import load_concrete_data
from src.models import train_traditional_models
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Train traditional ML models for concrete strength prediction'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to concrete dataset CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./models/ml_models',
        help='Output directory for models'
    )
    
    args = parser.parse_args()
    
    print("Loading data...")
    dataset = load_concrete_data(args.data_path)
    X_train, X_test, y_train, y_test = dataset.prepare_train_test_split()
    
    print(f"\nTraining traditional ML models...")
    ml_models, results = train_traditional_models(
        X_train, y_train, X_test, y_test
    )
    
    print(f"\nSaving models to {args.output_dir}...")
    ml_models.save_models(args.output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
