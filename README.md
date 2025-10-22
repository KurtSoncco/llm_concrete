# LLM Concrete Mix Design

A project for concrete mix design using Large Language Models (LLMs) with Supervised Fine-Tuning (SFT) and comparison with traditional Machine Learning models.

## Overview

This project implements an end-to-end pipeline for predicting concrete compressive strength using:
- **LLM with SFT**: Fine-tuned language models using LoRA for parameter-efficient training
- **Traditional ML Models**: Random Forest, XGBoost, Gradient Boosting, Ridge, Lasso, and MLP
- **Comprehensive Evaluation**: Side-by-side comparison of all models

## Features

- 🧠 **LLM Fine-Tuning**: Uses Supervised Fine-Tuning (SFT) with LoRA for efficient training
- 📊 **Multiple ML Models**: Includes 6 traditional ML models for comparison
- 🔬 **Synthetic Data Generation**: Built-in synthetic data generator for testing
- 📈 **Comprehensive Evaluation**: Detailed metrics (RMSE, MAE, R²) and visualizations
- 🛠️ **Modular Design**: Easy to extend and customize
- 🎯 **Multi-task Learning**: Predict both compressive and tensile strength simultaneously
- 📋 **Real Data Support**: Load actual concrete mix design datasets

## Project Structure

```
llm_concrete/
├── src/
│   ├── data/               # Data loading and preprocessing
│   │   └── concrete_dataset.py
│   ├── models/             # Traditional ML models
│   │   └── traditional_ml.py
│   ├── training/           # LLM fine-tuning
│   │   └── llm_trainer.py
│   └── evaluation/         # Model comparison and evaluation
│       └── comparison.py
├── scripts/
│   ├── run_pipeline.py     # Complete pipeline
│   ├── train_ml_models.py  # Train only ML models
│   └── train_llm.py        # Train only LLM
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KurtSoncco/llm_concrete.git
cd llm_concrete
```

2. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

3. Install dependencies:
```bash
uv sync
```

Alternatively, for development:
```bash
uv sync --extra dev
```

### Why uv?

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management. uv is significantly faster than pip and provides better dependency resolution. If you prefer pip, you can still use `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Quick Start

### Run Complete Pipeline (Multi-task)

Run the entire pipeline including ML models, LLM fine-tuning, and comparison for both compression and tensile strength:

```bash
uv run python scripts/run_pipeline.py --multitask
```

With custom options:
```bash
uv run python scripts/run_pipeline.py \
    --multitask \
    --llm-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --epochs 3 \
    --batch-size 4 \
    --output-dir ./results
```

### Train Multi-task LLM Only

```bash
uv run python scripts/train_multitask_llm.py \
    --compression-path data/Data_Compresion_Concreto.csv \
    --tensile-path data/Data_Traccion_Concreto.csv \
    --epochs 3 \
    --output-dir ./models/multitask_llm
```

### Train Only ML Models

```bash
uv run python scripts/train_ml_models.py --output-dir ./models/ml_models
```

### Train Only LLM

```bash
uv run python scripts/train_llm.py \
    --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --epochs 3 \
    --output-dir ./models/llm_model
```

## Usage Examples

### Using Multi-task Data Module

```python
from src.data import load_concrete_data

# Load both compression and tensile data
dataset = load_concrete_data(
    compression_path='data/Data_Compresion_Concreto.csv',
    tensile_path='data/Data_Traccion_Concreto.csv'
)

# Prepare train/val/test split (70/15/15)
X_train, X_val, X_test, y_train, y_val, y_test = dataset.prepare_train_val_test_split()

# Prepare data for multi-task LLM fine-tuning
llm_data = dataset.prepare_multitask_llm_dataset()
```

### Training Traditional ML Models

```python
from src.models import train_traditional_models

ml_models, results = train_traditional_models(
    X_train, y_train, X_test, y_test
)

# Save models
ml_models.save_models('./models/ml_models')
```

### Multi-task LLM Fine-Tuning

```python
from src.training import LLMFineTuner
from src.evaluation import MultiTaskMetrics

# Initialize trainer
llm_trainer = LLMFineTuner(
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    use_lora=True
)

# Prepare multi-task dataset
train_dataset = llm_trainer.prepare_dataset(llm_train_data)
val_dataset = llm_trainer.prepare_dataset(llm_val_data)

# Train
llm_trainer.train(
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    output_dir='./models/multitask_llm',
    num_epochs=3
)

# Evaluate multi-task model
metrics_calculator = MultiTaskMetrics()
results = metrics_calculator.evaluate_multitask_llm(
    llm_trainer, X_test, y_test, dataset.feature_columns
)
```

### Making Multi-task Predictions

```python
# Compression strength prediction
compression_instruction = "Given the following concrete mix design parameters, predict the compressive strength in MPa."
input_text = "Water: 150.0 kg/m³, Cement: 350.0 kg/m³, Aggregates: 1200.0 kg/m³, Age: 28 days"
compression_response = llm_trainer.predict(compression_instruction, input_text)

# Tensile strength prediction
tensile_instruction = "Given the following concrete mix design parameters, predict the tensile strength in MPa."
tensile_response = llm_trainer.predict(tensile_instruction, input_text)

print(f"Compression: {compression_response}")
print(f"Tensile: {tensile_response}")
```

## Dataset

The project supports both real concrete data and synthetic data generation:

### Real Data
- **Compression Data**: `data/Data_Compresion_Concreto.csv` (212 samples)
- **Tensile Data**: `data/Data_Traccion_Concreto.csv` (79 samples)

### Synthetic Data Generator
Creates realistic concrete mix design data based on typical ranges:

- **Input Features**:
  - Water (kg/m³)
  - Cement (kg/m³)
  - Aggregates (kg/m³)
  - Superplasticizer (kg/m³)
  - Age (days)
  - Factor (kg/m³)

- **Targets**: 
  - Compressive Strength (MPa)
  - Tensile Strength (MPa)

You can also use your own CSV data by passing `--compression-path` and `--tensile-path` to the scripts.

## Models

### Traditional ML Models
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting framework
- **Gradient Boosting**: Boosting ensemble method
- **Ridge Regression**: Linear regression with L2 regularization
- **Lasso Regression**: Linear regression with L1 regularization
- **MLP**: Multi-layer Perceptron neural network

### LLM Models
- Supports any HuggingFace causal language model
- Default: TinyLlama-1.1B-Chat-v1.0 (efficient for local training)
- Uses LoRA for parameter-efficient fine-tuning
- Instruction-following format for predictions

## Evaluation Metrics

All models are evaluated using:
- **MSE** (Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better  
- **RMSE** (Root Mean Squared Error): Lower is better
- **R²** (R-squared): Higher is better (max 1.0)

For multi-task learning, metrics are computed separately for each task:
- Compression strength prediction metrics
- Tensile strength prediction metrics

Results include:
- Comparison table (`multitask_comparison.csv`)
- Task-specific metrics (`compression_metrics.csv`, `tensile_metrics.csv`)
- Metrics visualization (`multitask_comparison.png`)
- Predictions vs actual plots (`predictions_vs_actual.png`)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- scikit-learn 1.3+
- XGBoost 2.0+
- PEFT (for LoRA)
- See `requirements.txt` for full list

## GPU Support

The pipeline automatically uses GPU if available via CUDA. For CPU-only training:
- LLM training will be slower but functional
- Consider using smaller models or fewer epochs

## Command Line Options

### run_pipeline.py

```
--data-path        Path to CSV dataset (default: synthetic data)
--llm-model        HuggingFace model name
--skip-ml          Skip traditional ML training
--skip-llm         Skip LLM fine-tuning
--epochs           Number of LLM training epochs (default: 3)
--batch-size       Batch size for LLM training (default: 4)
--output-dir       Output directory (default: ./results)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{llm_concrete,
  title = {LLM Concrete Mix Design},
  author = {Kurt Soncco},
  year = {2025},
  url = {https://github.com/KurtSoncco/llm_concrete}
}
```

## Acknowledgments

- Built with HuggingFace Transformers
- Uses LoRA from PEFT library
- Traditional ML models from scikit-learn and XGBoost
