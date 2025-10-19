#!/usr/bin/env python3
"""
Simple example demonstrating the concrete mix design prediction workflow.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import load_concrete_data
from src.models import TraditionalMLModels


def main():
    print("=" * 70)
    print("Simple Example: Concrete Strength Prediction with ML")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n1. Loading synthetic concrete data...")
    dataset = load_concrete_data()
    print(f"   Dataset shape: {dataset.data.shape}")
    print(f"   Sample data:\n{dataset.data.head()}")
    
    # Step 2: Prepare train/test split
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = dataset.prepare_train_test_split()
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Step 3: Train a simple Random Forest model
    print("\n3. Training Random Forest model...")
    ml_model = TraditionalMLModels()
    ml_model.train(X_train, y_train, models_to_train=['random_forest'])
    
    # Step 4: Evaluate
    print("\n4. Evaluating model...")
    results = ml_model.evaluate(X_test, y_test)
    
    # Step 5: Make a sample prediction
    print("\n5. Making sample prediction...")
    sample = X_test.iloc[0:1]
    prediction = ml_model.predict(sample, 'random_forest')
    actual = y_test.iloc[0]
    
    print(f"\n   Input features:")
    for col, val in sample.iloc[0].items():
        print(f"      {col}: {val:.2f}")
    print(f"\n   Predicted strength: {prediction[0]:.2f} MPa")
    print(f"   Actual strength: {actual:.2f} MPa")
    print(f"   Error: {abs(prediction[0] - actual):.2f} MPa")
    
    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
