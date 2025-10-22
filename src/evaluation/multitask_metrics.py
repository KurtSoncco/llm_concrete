"""
Multi-task evaluation utilities for LLM vs Traditional ML models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import re
import os


class MultiTaskMetrics:
    """
    Multi-task evaluation metrics for concrete strength prediction.
    """
    
    def __init__(self):
        """Initialize multi-task metrics calculator."""
        self.results = {}
        
    def extract_strength_from_text(self, text: str, task: str = None) -> Optional[float]:
        """
        Extract predicted strength value from LLM output text.
        
        Args:
            text: Generated text from LLM
            task: Task type ('compression' or 'tensile') for context
            
        Returns:
            Extracted strength value or None if not found
        """
        # More comprehensive patterns for strength extraction
        patterns = [
            # Direct MPa patterns
            r'(\d+\.?\d*)\s*MPa',
            r'(\d+\.?\d*)\s*mpa',
            r'(\d+\.?\d*)\s*MPA',
            
            # "is X" patterns
            r'is\s+(\d+\.?\d*)',
            r'=\s*(\d+\.?\d*)',
            r':\s*(\d+\.?\d*)',
            
            # Strength-specific patterns
            r'strength\s+is\s+(\d+\.?\d*)',
            r'strength\s+of\s+(\d+\.?\d*)',
            r'predicted.*?(\d+\.?\d*)\s*MPa',
            r'predicted.*?(\d+\.?\d*)',
            r'result.*?(\d+\.?\d*)',
            r'value.*?(\d+\.?\d*)',
            
            # Task-specific patterns
            r'compressive.*?(\d+\.?\d*)',
            r'tensile.*?(\d+\.?\d*)',
            
            # General number patterns
            r'(\d+\.?\d*)\s*units',
            r'(\d+\.?\d*)\s*kg',
        ]
        
        # Try each pattern
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    # Basic sanity check for concrete strength values
                    if task == 'compression' and 0 < value < 200:  # Reasonable compressive strength range
                        return value
                    elif task == 'tensile' and 0 < value < 20:    # Reasonable tensile strength range
                        return value
                    elif task is None and 0 < value < 200:       # General range
                        return value
                except:
                    continue
        
        # If no pattern found, try to find any reasonable number
        numbers = re.findall(r'\d+\.?\d*', text)
        for num_str in numbers:
            try:
                value = float(num_str)
                if 0 < value < 200:  # Reasonable range for concrete strength
                    return value
            except:
                continue
        
        return None
    
    def compute_metrics_per_task(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        task_name: str
    ) -> Dict[str, float]:
        """
        Compute MSE, MAE, RMSE, R² for a specific task.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            task_name: Name of the task (e.g., 'compression', 'tensile')
            
        Returns:
            Dictionary with metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {
                'mse': float('inf'),
                'mae': float('inf'),
                'rmse': float('inf'),
                'r2': -float('inf'),
                'n_samples': 0
            }
        
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n_samples': len(y_true_clean)
        }
    
    def evaluate_multitask_llm(
        self,
        llm_model,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        feature_columns: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate LLM model on both compression and tensile tasks.
        
        Args:
            llm_model: LLMFineTuner instance
            X_test: Test features
            y_test: Test targets (DataFrame with both compression and tensile columns)
            feature_columns: List of feature column names
            
        Returns:
            Dictionary with evaluation metrics for each task
        """
        print("Evaluating LLM model on multi-task...")
        
        compression_predictions = []
        tensile_predictions = []
        compression_valid_indices = []
        tensile_valid_indices = []
        
        # Compression task
        compression_instruction = (
            "Given the following concrete mix design parameters, "
            "predict the compressive strength in MPa."
        )
        
        # Tensile task
        tensile_instruction = (
            "Given the following concrete mix design parameters, "
            "predict the tensile strength in MPa."
        )
        
        for idx, row in X_test.iterrows():
            # Create input text
            input_parts = []
            if pd.notna(row['water']):
                input_parts.append(f"Water: {row['water']:.2f} kg/m³")
            if pd.notna(row['cement']):
                input_parts.append(f"Cement: {row['cement']:.2f} kg/m³")
            if pd.notna(row['aggregates']):
                input_parts.append(f"Aggregates: {row['aggregates']:.2f} kg/m³")
            if pd.notna(row['superplasticizer']):
                input_parts.append(f"Superplasticizer: {row['superplasticizer']:.2f} kg/m³")
            if pd.notna(row['age']):
                input_parts.append(f"Age: {row['age']} days")
            if pd.notna(row['factor']):
                input_parts.append(f"Factor: {row['factor']:.2f} kg/m³")
            
            input_text = ", ".join(input_parts)
            
            # Get the corresponding row from y_test using the same index
            y_row = y_test.loc[idx]
            
            # Predict compression strength
            if pd.notna(y_row['compressive_strength']):
                response = llm_model.predict(compression_instruction, input_text)
                predicted_value = self.extract_strength_from_text(response, 'compression')
                
                if predicted_value is not None:
                    compression_predictions.append(predicted_value)
                    compression_valid_indices.append(idx)
            
            # Predict tensile strength
            if pd.notna(y_row['tensile_strength']):
                response = llm_model.predict(tensile_instruction, input_text)
                predicted_value = self.extract_strength_from_text(response, 'tensile')
                
                if predicted_value is not None:
                    tensile_predictions.append(predicted_value)
                    tensile_valid_indices.append(idx)
        
        # Compute metrics for compression task
        compression_results = {}
        if len(compression_predictions) > 0:
            compression_y_true = y_test.loc[compression_valid_indices]['compressive_strength'].values
            compression_y_pred = np.array(compression_predictions)
            
            compression_results = self.compute_metrics_per_task(
                compression_y_true, compression_y_pred, 'compression'
            )
            
            print(f"Compression Task:")
            print(f"  Test RMSE: {compression_results['rmse']:.4f}, MAE: {compression_results['mae']:.4f}, R2: {compression_results['r2']:.4f}")
            print(f"  Valid predictions: {len(compression_predictions)}/{len(X_test)}")
        else:
            print("Warning: No valid compression predictions extracted from LLM")
            compression_results = {
                'mse': float('inf'),
                'mae': float('inf'),
                'rmse': float('inf'),
                'r2': -float('inf'),
                'n_samples': 0
            }
        
        # Compute metrics for tensile task
        tensile_results = {}
        if len(tensile_predictions) > 0:
            tensile_y_true = y_test.loc[tensile_valid_indices]['tensile_strength'].values
            tensile_y_pred = np.array(tensile_predictions)
            
            tensile_results = self.compute_metrics_per_task(
                tensile_y_true, tensile_y_pred, 'tensile'
            )
            
            print(f"Tensile Task:")
            print(f"  Test RMSE: {tensile_results['rmse']:.4f}, MAE: {tensile_results['mae']:.4f}, R2: {tensile_results['r2']:.4f}")
            print(f"  Valid predictions: {len(tensile_predictions)}/{len(X_test)}")
        else:
            print("Warning: No valid tensile predictions extracted from LLM")
            tensile_results = {
                'mse': float('inf'),
                'mae': float('inf'),
                'rmse': float('inf'),
                'r2': -float('inf'),
                'n_samples': 0
            }
        
        return {
            'compression': compression_results,
            'tensile': tensile_results
        }
    
    def evaluate_multitask_llm_fast(
        self,
        llm_model,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        feature_columns: List[str],
        max_samples: int = 50
    ) -> Dict[str, Dict[str, float]]:
        """
        Fast evaluation with sampling for large datasets.
        
        Args:
            llm_model: Trained LLM model
            X_test: Test features
            y_test: Test targets
            feature_columns: List of feature column names
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary with metrics for each task
        """
        print(f"Fast evaluation on {min(len(X_test), max_samples)} samples (sampled from {len(X_test)} total)")
        
        # Sample test data for faster evaluation
        if len(X_test) > max_samples:
            sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
            X_test_sample = X_test.iloc[sample_indices]
            y_test_sample = y_test.iloc[sample_indices]
        else:
            X_test_sample = X_test
            y_test_sample = y_test
        
        # Use the regular evaluation method on sampled data
        return self.evaluate_multitask_llm(llm_model, X_test_sample, y_test_sample, feature_columns)
    
    def generate_metrics_report(
        self,
        compression_metrics: Dict[str, float],
        tensile_metrics: Dict[str, float],
        output_dir: str = "./results"
    ) -> pd.DataFrame:
        """
        Generate comprehensive metrics report for both tasks.
        
        Args:
            compression_metrics: Metrics for compression task
            tensile_metrics: Metrics for tensile task
            output_dir: Directory to save results
            
        Returns:
            DataFrame with comparison metrics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comparison table
        comparison_data = []
        
        # Compression task
        comparison_data.append({
            'Task': 'Compression',
            'MSE': compression_metrics['mse'],
            'MAE': compression_metrics['mae'],
            'RMSE': compression_metrics['rmse'],
            'R²': compression_metrics['r2'],
            'N_Samples': compression_metrics['n_samples']
        })
        
        # Tensile task
        comparison_data.append({
            'Task': 'Tensile',
            'MSE': tensile_metrics['mse'],
            'MAE': tensile_metrics['mae'],
            'RMSE': tensile_metrics['rmse'],
            'R²': tensile_metrics['r2'],
            'N_Samples': tensile_metrics['n_samples']
        })
        
        df = pd.DataFrame(comparison_data)
        
        # Save individual task metrics
        compression_df = pd.DataFrame([compression_metrics])
        compression_df.to_csv(f"{output_dir}/compression_metrics.csv", index=False)
        
        tensile_df = pd.DataFrame([tensile_metrics])
        tensile_df.to_csv(f"{output_dir}/tensile_metrics.csv", index=False)
        
        # Save comparison
        df.to_csv(f"{output_dir}/multitask_comparison.csv", index=False)
        
        print(f"\nMulti-task Metrics Report:")
        print(df.to_string(index=False))
        print(f"\nResults saved to: {output_dir}/")
        
        return df
    
    def plot_multitask_comparison(
        self,
        comparison_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Create visualization comparing metrics across tasks.
        
        Args:
            comparison_df: DataFrame with comparison metrics
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        tasks = comparison_df['Task'].values
        metrics = ['MSE', 'MAE', 'RMSE', 'R²']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            bars = ax.bar(tasks, comparison_df[metric])
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            
            # Add value labels on bars
            for bar, value in zip(bars, comparison_df[metric]):
                if not np.isinf(value) and not np.isnan(value):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multi-task comparison plot saved to {save_path}")
        
        plt.close()
    
    def plot_predictions_vs_actual(
        self,
        y_true_compression: np.ndarray,
        y_pred_compression: np.ndarray,
        y_true_tensile: np.ndarray,
        y_pred_tensile: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot predicted vs actual values for both tasks.
        
        Args:
            y_true_compression: True compression values
            y_pred_compression: Predicted compression values
            y_true_tensile: True tensile values
            y_pred_tensile: Predicted tensile values
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Compression plot
        axes[0].scatter(y_true_compression, y_pred_compression, alpha=0.6)
        min_val = min(y_true_compression.min(), y_pred_compression.min())
        max_val = max(y_true_compression.max(), y_pred_compression.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0].set_xlabel('Actual Compressive Strength (MPa)')
        axes[0].set_ylabel('Predicted Compressive Strength (MPa)')
        axes[0].set_title('Compression Strength Prediction')
        axes[0].grid(True, alpha=0.3)
        
        # Tensile plot
        axes[1].scatter(y_true_tensile, y_pred_tensile, alpha=0.6)
        min_val = min(y_true_tensile.min(), y_pred_tensile.min())
        max_val = max(y_true_tensile.max(), y_pred_tensile.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1].set_xlabel('Actual Tensile Strength (MPa)')
        axes[1].set_ylabel('Predicted Tensile Strength (MPa)')
        axes[1].set_title('Tensile Strength Prediction')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions vs actual plot saved to {save_path}")
        
        plt.close()


def evaluate_multitask_models(
    llm_results: Dict[str, Dict[str, float]],
    ml_results: Optional[Dict[str, Dict[str, float]]] = None,
    output_dir: str = "./results"
) -> pd.DataFrame:
    """
    Convenience function to evaluate and compare multi-task models.
    
    Args:
        llm_results: Results from LLM model
        ml_results: Optional results from traditional ML models
        output_dir: Directory to save results
        
    Returns:
        Comparison DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_calculator = MultiTaskMetrics()
    
    # Generate LLM metrics report
    llm_comparison = metrics_calculator.generate_metrics_report(
        llm_results['compression'],
        llm_results['tensile'],
        output_dir
    )
    
    # Plot comparison
    metrics_calculator.plot_multitask_comparison(
        llm_comparison,
        save_path=f"{output_dir}/multitask_comparison.png"
    )
    
    return llm_comparison
