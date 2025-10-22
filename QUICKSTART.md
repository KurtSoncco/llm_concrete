# Quick Start Guide

This guide will help you get started with the LLM Concrete Mix Design project quickly.

## Installation

### 1. Clone and Setup

```bash
git clone https://github.com/KurtSoncco/llm_concrete.git
cd llm_concrete

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Dependencies

#### For ML-only usage (lightweight):
```bash
uv sync --extra ml-only
```

#### For full LLM capabilities:
```bash
uv sync
```

## Quick Examples

### Example 1: Simple ML Model

Run a simple Random Forest model on synthetic data:

```bash
uv run python scripts/example_simple.py
```

**Output:**
- Trains a Random Forest model
- Makes predictions on test data
- Shows evaluation metrics (RMSE, MAE, R²)

### Example 2: Train All ML Models

Train and compare 6 different ML models:

```bash
uv run python scripts/train_ml_models.py
```

**Models trained:**
- Random Forest
- XGBoost
- Gradient Boosting
- Ridge Regression
- Lasso Regression
- MLP Neural Network

### Example 3: Fine-tune an LLM

Train an LLM for concrete strength prediction:

```bash
uv run python scripts/train_llm.py --epochs 3 --batch-size 4
```

**Note:** Requires GPU for reasonable training times. CPU training is possible but slow.

### Example 4: Complete Pipeline

Run the entire pipeline (ML + LLM + Comparison):

```bash
uv run python scripts/run_pipeline.py
```

**Note:** This takes significant time and resources. Consider using:
```bash
uv run python scripts/run_pipeline.py --skip-llm  # Train only ML models
```

## Testing Your Installation

Run the test suite to verify everything is working:

```bash
uv run python scripts/test_implementation.py
```

## Using Your Own Data

Prepare a CSV file with these columns:
- cement
- blast_furnace_slag
- fly_ash
- water
- superplasticizer
- coarse_aggregate
- fine_aggregate
- age
- compressive_strength (target)

Then run:

```bash
uv run python scripts/run_pipeline.py --data-path /path/to/your/data.csv
```

## Common Issues

### 1. Out of Memory (LLM Training)

**Solution:** Reduce batch size
```bash
uv run python scripts/train_llm.py --batch-size 2
```

### 2. No GPU Available

**Solution:** LLM training will use CPU (slower). Consider:
- Using a smaller model
- Reducing epochs
- Using only ML models with `--skip-llm`

### 3. Import Errors

**Solution:** Ensure dependencies are installed:
```bash
uv sync
```

## Understanding the Results

### Evaluation Metrics

- **RMSE** (Root Mean Squared Error): Lower is better. Measures average prediction error.
- **MAE** (Mean Absolute Error): Lower is better. Average absolute difference from actual values.
- **R²** (R-squared): Higher is better (max 1.0). Proportion of variance explained by the model.

### Output Files

After running the pipeline, check the `results/` directory:
- `model_comparison.csv` - Table comparing all models
- `comparison_metrics.png` - Visual comparison chart
- `predictions_vs_actual.png` - Scatter plots of predictions

## Next Steps

1. **Experiment with hyperparameters** - Modify `config.yaml`
2. **Try different LLM models** - Use `--llm-model` parameter
3. **Add your own data** - Replace synthetic data with real concrete mix designs
4. **Extend the models** - Add new ML algorithms in `src/models/`

## Getting Help

- Check the main README.md for detailed documentation
- Review the example scripts in `scripts/`
- Open an issue on GitHub for bugs or questions

## Performance Tips

### For Faster Training:
1. Use GPU for LLM training
2. Reduce `--epochs` for quick experiments
3. Use `--skip-llm` to test ML models only
4. Start with synthetic data (faster to generate)

### For Better Results:
1. Use more training epochs (5-10)
2. Try different LLM models (larger models often perform better)
3. Tune hyperparameters in `config.yaml`
4. Use real concrete mix design data

## Example Workflow

Here's a recommended workflow for new users:

```bash
# 1. Test installation
uv run python scripts/test_implementation.py

# 2. Run simple example
uv run python scripts/example_simple.py

# 3. Train ML models
uv run python scripts/train_ml_models.py

# 4. (Optional) Train LLM if you have GPU
uv run python scripts/train_llm.py --epochs 3

# 5. Compare all models (or skip LLM if not trained)
uv run python scripts/run_pipeline.py --skip-llm
```

Enjoy using the LLM Concrete Mix Design system! 🚀
