# Understanding LLM Fine-Tuning for Concrete Mix Design

## What is Supervised Fine-Tuning (SFT)?

Supervised Fine-Tuning (SFT) is a technique where we take a pre-trained language model and further train it on domain-specific data to specialize it for our task. In this project, we're teaching an LLM to predict concrete compressive strength based on mix design parameters.

## Why Use LLMs for Concrete Strength Prediction?

### Advantages:
1. **Natural Language Interface**: Can understand queries in plain English
2. **Interpretability**: Generates human-readable explanations
3. **Transfer Learning**: Leverages knowledge from pre-training
4. **Flexibility**: Can be easily adapted to related tasks (e.g., tensile strength)
5. **Few-Shot Learning**: May generalize better with limited data

### Challenges:
1. **Computational Resources**: Requires GPU for efficient training
2. **Output Parsing**: Numerical values must be extracted from text
3. **Consistency**: May have higher variance than traditional models
4. **Training Time**: Longer than traditional ML models

## How Our SFT Implementation Works

### 1. Data Format Transformation

We convert tabular data into instruction-following format:

**Before (Tabular):**
```
cement  | slag | fly_ash | ... | strength
540.0   | 0.0  | 0.0     | ... | 79.99
```

**After (Instruction Format):**
```
### Instruction:
Given the following concrete mix design parameters, predict the compressive strength in MPa.

### Input:
Cement: 540.0 kg/m³, Blast Furnace Slag: 0.0 kg/m³, Fly Ash: 0.0 kg/m³, 
Water: 162.0 kg/m³, Superplasticizer: 2.5 kg/m³, Coarse Aggregate: 1040.0 kg/m³, 
Fine Aggregate: 676.0 kg/m³, Age: 28 days

### Response:
The predicted compressive strength is 79.99 MPa.
```

### 2. LoRA (Low-Rank Adaptation)

We use LoRA for parameter-efficient fine-tuning:

**What is LoRA?**
- Freezes the original model weights
- Adds small trainable adapter layers
- Reduces trainable parameters by ~90%
- Maintains model quality

**Configuration:**
```python
lora_config = LoraConfig(
    r=8,                # Rank of adaptation
    lora_alpha=16,      # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
```

### 3. Training Process

```python
# 1. Load pre-trained model
llm_trainer = LLMFineTuner(
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0'
)

# 2. Prepare instruction-formatted data
train_dataset = llm_trainer.prepare_dataset(llm_data)

# 3. Fine-tune with SFT
llm_trainer.train(
    train_dataset=train_dataset,
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4
)
```

## Model Selection

### TinyLlama-1.1B (Default)
- **Size**: 1.1 billion parameters
- **Advantages**: Fast training, runs on consumer GPUs
- **Best for**: Testing, prototyping, resource-constrained environments

### Alternative Models

#### Small Models (< 3B parameters)
```python
llm_trainer = LLMFineTuner(model_name='microsoft/phi-2')  # 2.7B
llm_trainer = LLMFineTuner(model_name='stabilityai/stablelm-2-1_6b')  # 1.6B
```

#### Medium Models (3-7B parameters)
```python
llm_trainer = LLMFineTuner(model_name='mistralai/Mistral-7B-v0.1')  # 7B
llm_trainer = LLMFineTuner(model_name='meta-llama/Llama-2-7b')  # 7B
```

#### Large Models (> 7B parameters)
```python
llm_trainer = LLMFineTuner(model_name='meta-llama/Llama-2-13b')  # 13B
# Note: Requires significant GPU memory (24GB+)
```

## Hardware Requirements

### Minimum (CPU Only)
- **RAM**: 8GB
- **Training Time**: Very slow (hours per epoch)
- **Recommended**: Use TinyLlama or smaller
- **Batch Size**: 1-2

### Recommended (GPU)
- **GPU**: NVIDIA with 8GB+ VRAM (e.g., RTX 3060, T4)
- **RAM**: 16GB
- **Training Time**: ~30 min per epoch (TinyLlama)
- **Batch Size**: 4-8

### Optimal (High-end GPU)
- **GPU**: NVIDIA with 16GB+ VRAM (e.g., RTX 4090, A100)
- **RAM**: 32GB+
- **Training Time**: ~10 min per epoch (TinyLlama)
- **Batch Size**: 8-16
- **Can use larger models**: Mistral-7B, Llama-2-7B

## Training Tips

### 1. Start Small
```bash
# Begin with a small model and few epochs
python scripts/train_llm.py --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --epochs 1
```

### 2. Monitor Training
The training script logs:
- Training loss
- Evaluation loss (if eval data provided)
- Training steps

Lower loss = better fit to training data

### 3. Adjust Batch Size
```bash
# If out of memory, reduce batch size
python scripts/train_llm.py --batch-size 2

# If training is slow, increase batch size
python scripts/train_llm.py --batch-size 8
```

### 4. Hyperparameter Tuning
Edit `config.yaml`:
```yaml
llm:
  training:
    num_epochs: 5          # More epochs = better fit
    batch_size: 4          # Adjust based on GPU
    learning_rate: 0.0002  # Lower = more stable
    warmup_steps: 50       # Gradual learning rate increase
```

## Prediction and Inference

### Making Predictions

```python
instruction = "Given the following concrete mix design parameters, predict the compressive strength in MPa."

input_text = """Cement: 540.0 kg/m³, Blast Furnace Slag: 0.0 kg/m³, 
Fly Ash: 0.0 kg/m³, Water: 162.0 kg/m³, Superplasticizer: 2.5 kg/m³, 
Coarse Aggregate: 1040.0 kg/m³, Fine Aggregate: 676.0 kg/m³, Age: 28 days"""

response = llm_trainer.predict(instruction, input_text)
print(response)
# Output: "The predicted compressive strength is 79.99 MPa."
```

### Extracting Numerical Values

The evaluation module automatically extracts numbers:

```python
from src.evaluation import ModelComparison

comparator = ModelComparison()
value = comparator.extract_strength_from_text(response)
print(value)  # 79.99
```

## Comparing LLM vs Traditional ML

### Performance Expectations

**Traditional ML (XGBoost, Random Forest):**
- RMSE: ~3-6 MPa on test set
- R²: 0.80-0.95
- Training time: Seconds to minutes
- Inference: Milliseconds per prediction

**Fine-tuned LLM:**
- RMSE: ~5-15 MPa (typically higher variance)
- R²: 0.70-0.90 (depends on model size and training)
- Training time: Minutes to hours
- Inference: 1-5 seconds per prediction

### When to Use Each

**Use Traditional ML when:**
- Speed is critical
- Resources are limited
- You need guaranteed numerical output
- You want the best accuracy for tabular data

**Use LLM when:**
- You need natural language interface
- Interpretability is important
- You want to handle multiple related tasks
- You need explanations with predictions

## Advanced Topics

### 1. Multi-Task Learning

Extend to predict both compressive and tensile strength:

```python
# Modify prepare_llm_dataset to include both tasks
output_text = f"""
The predicted compressive strength is {row['compressive_strength']:.2f} MPa.
The predicted tensile strength is {row['tensile_strength']:.2f} MPa.
"""
```

### 2. Few-Shot Prompting

Before fine-tuning, try few-shot learning:

```python
prompt = """
Example 1: Cement: 400kg/m³, ... → Strength: 45.5 MPa
Example 2: Cement: 500kg/m³, ... → Strength: 62.3 MPa

Now predict for: Cement: 540kg/m³, ...
"""
```

### 3. Quantization

For faster inference and lower memory:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # 8-bit quantization
    device_map='auto'
)
```

### 4. Distributed Training

For very large models:

```python
accelerate launch scripts/train_llm.py
```

## Troubleshooting

### Issue: Out of Memory
**Solutions:**
1. Reduce batch size: `--batch-size 1`
2. Use gradient checkpointing (edit training code)
3. Use a smaller model
4. Enable 8-bit loading

### Issue: Poor Predictions
**Solutions:**
1. Train for more epochs: `--epochs 5`
2. Use more training data
3. Check learning rate (try 1e-4 or 5e-5)
4. Try a larger model

### Issue: Slow Training
**Solutions:**
1. Use GPU instead of CPU
2. Increase batch size if memory allows
3. Reduce training data size for testing
4. Use a smaller model for prototyping

## Resources

- **Transformers Docs**: https://huggingface.co/docs/transformers
- **PEFT/LoRA**: https://github.com/huggingface/peft
- **Model Hub**: https://huggingface.co/models

## Conclusion

LLM fine-tuning with SFT offers a novel approach to concrete strength prediction. While traditional ML models currently outperform LLMs in pure accuracy for this tabular task, LLMs provide unique advantages in interpretability and natural language interaction. This implementation gives you the tools to experiment with both approaches and choose the best solution for your needs.
