"""
Example Usage Patterns for LLM Concrete Mix Design

This file demonstrates various ways to use the library components.
"""

# ==============================================================================
# Example 1: Load Data and Explore
# ==============================================================================

from src.data import load_concrete_data

# Load synthetic data
dataset = load_concrete_data()
print(f"Dataset size: {len(dataset.data)}")
print(f"\nFirst 5 samples:")
print(dataset.data.head())

# Or load your own CSV
# dataset = load_concrete_data('/path/to/your/data.csv')


# ==============================================================================
# Example 2: Train a Single ML Model
# ==============================================================================

from src.data import load_concrete_data
from src.models import TraditionalMLModels

# Load and split data
dataset = load_concrete_data()
X_train, X_test, y_train, y_test = dataset.prepare_train_test_split()

# Train just Random Forest
ml_models = TraditionalMLModels()
ml_models.train(X_train, y_train, models_to_train=['random_forest'])

# Evaluate
results = ml_models.evaluate(X_test, y_test)
print(f"Random Forest RMSE: {results['random_forest']['test_rmse']:.2f}")


# ==============================================================================
# Example 3: Train All ML Models and Compare
# ==============================================================================

from src.models import train_traditional_models

# This is a convenience function that trains and evaluates all models
ml_models, results = train_traditional_models(X_train, y_train, X_test, y_test)

# Find best model
best_model = min(results.items(), key=lambda x: x[1]['test_rmse'])
print(f"Best model: {best_model[0]} with RMSE: {best_model[1]['test_rmse']:.2f}")


# ==============================================================================
# Example 4: Make Predictions with ML Models
# ==============================================================================

# Single prediction
sample = X_test.iloc[0:1]
prediction = ml_models.predict(sample, 'xgboost')
print(f"Predicted strength: {prediction[0]:.2f} MPa")

# Batch predictions
predictions = ml_models.predict(X_test, 'random_forest')
print(f"Predicted {len(predictions)} samples")


# ==============================================================================
# Example 5: Save and Load ML Models
# ==============================================================================

# Save trained models
ml_models.save_models('./my_models')

# Later, load them back
new_ml_models = TraditionalMLModels()
new_ml_models.load_models('./my_models')

# Use loaded models
predictions = new_ml_models.predict(X_test, 'random_forest')


# ==============================================================================
# Example 6: Prepare Data for LLM Fine-Tuning
# ==============================================================================

from src.data import load_concrete_data

dataset = load_concrete_data()
llm_data = dataset.prepare_llm_dataset()

# Examine the format
print("Example LLM training sample:")
print(f"Instruction: {llm_data[0]['instruction']}")
print(f"Input: {llm_data[0]['input']}")
print(f"Output: {llm_data[0]['output']}")

# Split for training and evaluation
train_size = int(0.8 * len(llm_data))
llm_train = llm_data[:train_size]
llm_eval = llm_data[train_size:]


# ==============================================================================
# Example 7: Fine-tune an LLM (requires torch/transformers)
# ==============================================================================

try:
    from src.training import LLMFineTuner
    
    # Initialize trainer with a small model
    llm_trainer = LLMFineTuner(
        model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        use_lora=True
    )
    
    # Prepare datasets
    train_dataset = llm_trainer.prepare_dataset(llm_train)
    eval_dataset = llm_trainer.prepare_dataset(llm_eval)
    
    # Train
    llm_trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir='./my_llm_model',
        num_epochs=3,
        batch_size=4
    )
    
except ImportError:
    print("LLM training requires: pip install torch transformers peft accelerate")


# ==============================================================================
# Example 8: Make Predictions with Fine-tuned LLM
# ==============================================================================

try:
    from src.training import LLMFineTuner
    
    # Load or use trained model
    llm_trainer = LLMFineTuner(model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    # llm_trainer.load_model('./my_llm_model')  # If you saved it
    
    # Prepare input
    instruction = "Given the following concrete mix design parameters, predict the compressive strength in MPa."
    input_text = """Cement: 540.0 kg/m³, Blast Furnace Slag: 0.0 kg/m³, 
    Fly Ash: 0.0 kg/m³, Water: 162.0 kg/m³, Superplasticizer: 2.5 kg/m³, 
    Coarse Aggregate: 1040.0 kg/m³, Fine Aggregate: 676.0 kg/m³, Age: 28 days"""
    
    # Get prediction
    response = llm_trainer.predict(instruction, input_text)
    print(f"LLM Response: {response}")
    
except ImportError:
    print("LLM inference requires: pip install torch transformers peft")


# ==============================================================================
# Example 9: Compare ML and LLM Models
# ==============================================================================

try:
    from src.evaluation import ModelComparison, compare_all_models
    
    # Assuming you have ml_results and llm_results from previous steps
    # ml_results = {...}  # From training ML models
    # llm_results = {...}  # From evaluating LLM
    
    # Create comparison
    comparator = ModelComparison()
    comparison_df = comparator.compare_models(ml_results, llm_results)
    
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Generate visualizations
    comparator.plot_comparison(comparison_df, save_path='./comparison.png')
    
except Exception as e:
    print(f"Comparison requires both ML and LLM results: {e}")


# ==============================================================================
# Example 10: Custom Configuration
# ==============================================================================

import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use configuration
model_config = config['ml_models']['random_forest']
print(f"Random Forest config: {model_config}")

# Modify and use
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=model_config['n_estimators'],
    max_depth=model_config['max_depth'],
    random_state=model_config['random_state']
)


# ==============================================================================
# Example 11: Batch Processing Multiple Samples
# ==============================================================================

import pandas as pd
from src.models import TraditionalMLModels

# Create multiple samples
samples = pd.DataFrame({
    'cement': [540.0, 400.0, 300.0],
    'blast_furnace_slag': [0.0, 100.0, 50.0],
    'fly_ash': [0.0, 0.0, 100.0],
    'water': [162.0, 180.0, 200.0],
    'superplasticizer': [2.5, 5.0, 0.0],
    'coarse_aggregate': [1040.0, 1000.0, 950.0],
    'fine_aggregate': [676.0, 700.0, 750.0],
    'age': [28, 28, 28]
})

# Load trained model
ml_models = TraditionalMLModels()
ml_models.load_models('./my_models')

# Predict all at once
predictions = ml_models.predict(samples, 'xgboost')

for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: {pred:.2f} MPa")


# ==============================================================================
# Example 12: Error Handling and Validation
# ==============================================================================

from src.data import ConcreteDataset

try:
    # Try to load non-existent file
    dataset = ConcreteDataset('nonexistent.csv')
    dataset.load_data()
except FileNotFoundError:
    print("File not found, using synthetic data instead")
    dataset = ConcreteDataset()
    dataset.load_data()

# Validate data
assert len(dataset.data) > 0, "Dataset is empty"
assert all(col in dataset.data.columns for col in dataset.feature_columns), \
    "Missing required columns"

print("Data validation passed!")


# ==============================================================================
# Example 13: Cross-validation (custom implementation)
# ==============================================================================

from sklearn.model_selection import cross_val_score
from src.models import TraditionalMLModels

ml_models = TraditionalMLModels()
X = dataset.data[dataset.feature_columns]
y = dataset.data['compressive_strength']

# Get a specific model
rf_model = ml_models.models['random_forest']

# Perform cross-validation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scores = cross_val_score(rf_model, X_scaled, y, cv=5, 
                         scoring='neg_root_mean_squared_error')
print(f"Cross-validation RMSE: {-scores.mean():.2f} (+/- {scores.std():.2f})")


# ==============================================================================
# Example 14: Feature Importance Analysis
# ==============================================================================

import matplotlib.pyplot as plt
from src.models import TraditionalMLModels

# Train model
ml_models = TraditionalMLModels()
ml_models.train(X_train, y_train, models_to_train=['random_forest'])

# Get feature importance
rf_model = ml_models.trained_models['random_forest']
importances = rf_model.feature_importances_

# Plot
plt.figure(figsize=(10, 6))
plt.barh(dataset.feature_columns, importances)
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.savefig('./feature_importance.png')
print("Feature importance plot saved")


# ==============================================================================
# Example 15: Pipeline with Custom Preprocessing
# ==============================================================================

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

# Create custom pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)
print(f"Custom pipeline - RMSE: {rmse:.2f}, R²: {r2:.4f}")


print("\n" + "="*70)
print("All examples completed!")
print("="*70)
