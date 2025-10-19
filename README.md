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

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Run Complete Pipeline

Run the entire pipeline including ML models, LLM fine-tuning, and comparison:

```bash
python scripts/run_pipeline.py
```

With custom options:
```bash
python scripts/run_pipeline.py \
    --llm-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --epochs 3 \
    --batch-size 4 \
    --output-dir ./results
```

### Train Only ML Models

```bash
python scripts/train_ml_models.py --output-dir ./models/ml_models
```

### Train Only LLM

```bash
python scripts/train_llm.py \
    --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --epochs 3 \
    --output-dir ./models/llm_model
```

## Usage Examples

### Using the Data Module

```python
from src.data import load_concrete_data

# Load synthetic data
dataset = load_concrete_data()

# Prepare train/test split
X_train, X_test, y_train, y_test = dataset.prepare_train_test_split()

# Prepare data for LLM fine-tuning
llm_data = dataset.prepare_llm_dataset()
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

### Fine-Tuning LLM

```python
from src.training import LLMFineTuner

# Initialize trainer
llm_trainer = LLMFineTuner(
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    use_lora=True
)

# Prepare dataset
train_dataset = llm_trainer.prepare_dataset(llm_train_data)

# Train
llm_trainer.train(
    train_dataset=train_dataset,
    output_dir='./models/llm_model',
    num_epochs=3
)
```

### Making Predictions

```python
# With LLM
instruction = "Given the following concrete mix design parameters, predict the compressive strength in MPa."
input_text = "Cement: 540.0 kg/m³, Blast Furnace Slag: 0.0 kg/m³, ..."
response = llm_trainer.predict(instruction, input_text)

# With ML model
predictions = ml_models.predict(X_test, model_name='random_forest')
```

## Dataset

The project includes a synthetic data generator that creates realistic concrete mix design data based on typical ranges:

- **Input Features**:
  - Cement (kg/m³)
  - Blast Furnace Slag (kg/m³)
  - Fly Ash (kg/m³)
  - Water (kg/m³)
  - Superplasticizer (kg/m³)
  - Coarse Aggregate (kg/m³)
  - Fine Aggregate (kg/m³)
  - Age (days)

- **Target**: Compressive Strength (MPa)

You can also use your own CSV data by passing `--data-path` to the scripts.

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
- **RMSE** (Root Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better
- **R²** (R-squared): Higher is better (max 1.0)

Results include:
- Comparison table (`model_comparison.csv`)
- Metrics visualization (`comparison_metrics.png`)
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
