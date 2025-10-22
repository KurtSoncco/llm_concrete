# Multi-task LLM Guide for Concrete Strength Prediction

## Overview

This guide explains how to use the multi-task LLM fine-tuning system for predicting both compressive and tensile strength of concrete from mix design parameters.

## What is Multi-task Learning?

Multi-task learning trains a single model to perform multiple related tasks simultaneously. In our case:
- **Task 1**: Predict compressive strength from concrete mix parameters
- **Task 2**: Predict tensile strength from concrete mix parameters

### Benefits:
- **Shared Knowledge**: The model learns common patterns across both tasks
- **Better Generalization**: Training on multiple tasks often improves performance
- **Efficiency**: One model handles both predictions instead of two separate models

## Data Structure

### Compression Data (Data_Compresion_Concreto.csv)
- **Features**: w (water), c (cement), sf, fa, sp (superplasticizer), sin_fa, ex_cl, od, age
- **Target**: fcr (compressive strength in MPa)
- **Samples**: 212

### Tensile Data (Data_Traccion_Concreto.csv)
- **Features**: Agua (water), Agregados (aggregates), Cemento (cement), Factor
- **Target**: Resistencia a Tracción (tensile strength in MPa)
- **Samples**: 79

### Unified Schema
The system maps both datasets to a unified feature space:
- water, cement, aggregates, superplasticizer, age, factor
- compressive_strength, tensile_strength

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Run Multi-task Training
```bash
uv run python scripts/train_multitask_llm.py
```

### 3. Run Complete Pipeline
```bash
uv run python scripts/run_pipeline.py --multitask
```

## Usage Examples

### Basic Multi-task Training
```python
from src.data import load_concrete_data
from src.training import LLMFineTuner
from src.evaluation import MultiTaskMetrics

# Load multi-task data
dataset = load_concrete_data(
    compression_path='data/Data_Compresion_Concreto.csv',
    tensile_path='data/Data_Traccion_Concreto.csv'
)

# Prepare train/val/test split (70/15/15)
X_train, X_val, X_test, y_train, y_val, y_test = dataset.prepare_train_val_test_split()

# Prepare multi-task LLM dataset
llm_data = dataset.prepare_multitask_llm_dataset()

# Initialize LLM trainer
llm_trainer = LLMFineTuner(
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    use_lora=True
)

# Train
train_dataset = llm_trainer.prepare_dataset(llm_data[:int(0.7 * len(llm_data))])
val_dataset = llm_trainer.prepare_dataset(llm_data[int(0.7 * len(llm_data)):int(0.85 * len(llm_data))])

llm_trainer.train(
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    output_dir='./models/multitask_llm',
    num_epochs=3
)
```

### Making Predictions
```python
# Compression prediction
compression_instruction = "Given the following concrete mix design parameters, predict the compressive strength in MPa."
input_text = "Water: 150.0 kg/m³, Cement: 350.0 kg/m³, Aggregates: 1200.0 kg/m³, Age: 28 days"

compression_response = llm_trainer.predict(compression_instruction, input_text)
print(f"Compression: {compression_response}")

# Tensile prediction
tensile_instruction = "Given the following concrete mix design parameters, predict the tensile strength in MPa."
tensile_response = llm_trainer.predict(tensile_instruction, input_text)
print(f"Tensile: {tensile_response}")
```

### Evaluation
```python
# Evaluate multi-task model
metrics_calculator = MultiTaskMetrics()
results = metrics_calculator.evaluate_multitask_llm(
    llm_trainer,
    X_test,
    y_test,
    dataset.feature_columns
)

# Generate report
comparison_df = metrics_calculator.generate_metrics_report(
    results['compression'],
    results['tensile'],
    output_dir='./results'
)
```

## Configuration

### config.yaml
```yaml
# Multi-task configuration
multitask:
  tasks: ["compression", "tensile"]
  task_weights: [1.0, 1.0]  # Equal weighting
  compression_instruction: "Given the following concrete mix design parameters, predict the compressive strength in MPa."
  tensile_instruction: "Given the following concrete mix design parameters, predict the tensile strength in MPa."

# Data splits
data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

## Training Process

### 1. Data Preparation
- Load both compression and tensile datasets
- Map to unified feature schema
- Handle missing values (NaN for unavailable features/targets)
- Split into train/val/test (70/15/15)

### 2. Instruction Formatting
Each training example includes:
- **Instruction**: Task-specific prompt
- **Input**: Concrete mix parameters
- **Output**: Predicted strength value
- **Task**: "compression" or "tensile"

### 3. Training
- Use LoRA for parameter-efficient fine-tuning
- Train on mixed compression and tensile examples
- Validate on both tasks during training

### 4. Evaluation
- Compute MSE, MAE, RMSE, R² for each task separately
- Generate comparison visualizations
- Save metrics to CSV files

## Output Files

After training, you'll find:
- `results/compression_metrics.csv` - Metrics for compression task
- `results/tensile_metrics.csv` - Metrics for tensile task
- `results/multitask_comparison.csv` - Combined comparison
- `results/multitask_comparison.png` - Visualization
- `models/multitask_llm/` - Saved fine-tuned model

## Metrics Interpretation

### MSE (Mean Squared Error)
- Lower is better
- Penalizes large errors more heavily
- Units: (MPa)²

### MAE (Mean Absolute Error)
- Lower is better
- Average absolute difference
- Units: MPa

### RMSE (Root Mean Squared Error)
- Lower is better
- Same units as target variable
- Units: MPa

### R² (R-squared)
- Higher is better (max 1.0)
- Proportion of variance explained
- Unitless

## Troubleshooting

### Issue: Poor Performance on One Task
**Solutions:**
1. Adjust task weights in config.yaml
2. Increase training epochs
3. Check data quality for that task
4. Try different model architecture

### Issue: Out of Memory
**Solutions:**
1. Reduce batch size: `--batch-size 2`
2. Use smaller model: `--model-name microsoft/phi-2`
3. Reduce max_length in config

### Issue: Inconsistent Predictions
**Solutions:**
1. Increase training epochs
2. Check instruction format consistency
3. Verify data preprocessing
4. Use temperature=0.1 for more deterministic outputs

## Advanced Usage

### Custom Task Weights
```yaml
multitask:
  task_weights: [2.0, 1.0]  # Weight compression 2x more than tensile
```

### Different Model Sizes
```bash
# Small model (faster training)
uv run python scripts/train_multitask_llm.py --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Medium model (better performance)
uv run python scripts/train_multitask_llm.py --model-name microsoft/phi-2

# Large model (best performance, requires more resources)
uv run python scripts/train_multitask_llm.py --model-name mistralai/Mistral-7B-v0.1
```

### Custom Instructions
Modify the instructions in config.yaml to change how the model interprets tasks:
```yaml
multitask:
  compression_instruction: "Calculate the compressive strength of concrete with these mix parameters:"
  tensile_instruction: "Determine the tensile strength of concrete given these components:"
```

## Performance Expectations

### Typical Results
- **Compression Task**: RMSE 3-8 MPa, R² 0.7-0.9
- **Tensile Task**: RMSE 0.5-1.5 MPa, R² 0.6-0.8

### Factors Affecting Performance
1. **Model Size**: Larger models generally perform better
2. **Training Data**: More samples improve performance
3. **Feature Quality**: Complete feature information helps
4. **Task Balance**: Balanced training examples across tasks

## Future Enhancements

1. **Additional Tasks**: Add flexural strength, durability prediction
2. **Task-specific Heads**: Separate output heads for each task
3. **Transfer Learning**: Pre-train on synthetic data, fine-tune on real data
4. **Uncertainty Quantification**: Add confidence intervals to predictions
5. **Cross-validation**: Implement k-fold validation for robust evaluation

## Resources

- [Multi-task Learning Survey](https://arxiv.org/abs/1706.05098)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
