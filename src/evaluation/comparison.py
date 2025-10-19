"""
Evaluation and comparison utilities for LLM vs Traditional ML models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import re


class ModelComparison:
    """
    Compare LLM and traditional ML models for concrete strength prediction.
    """
    
    def __init__(self):
        """Initialize model comparison."""
        self.results = {}
        
    def extract_strength_from_text(self, text: str) -> Optional[float]:
        """
        Extract predicted strength value from LLM output text.
        
        Args:
            text: Generated text from LLM
            
        Returns:
            Extracted strength value or None if not found
        """
        # Look for patterns like "XX.XX MPa" or "is XX.XX"
        patterns = [
            r'(\d+\.?\d*)\s*MPa',
            r'is\s+(\d+\.?\d*)',
            r'strength\s+is\s+(\d+\.?\d*)',
            r'predicted.*?(\d+\.?\d*)\s*MPa'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except:
                    continue
        
        # If no pattern found, try to find any number
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[0])
            except:
                pass
        
        return None
    
    def evaluate_llm(
        self,
        llm_model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_columns: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate LLM model on test set.
        
        Args:
            llm_model: LLMFineTuner instance
            X_test: Test features
            y_test: True target values
            feature_columns: List of feature column names
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating LLM model...")
        predictions = []
        valid_indices = []
        
        instruction = (
            "Given the following concrete mix design parameters, "
            "predict the compressive strength in MPa."
        )
        
        for idx, row in X_test.iterrows():
            input_text = (
                f"Cement: {row['cement']:.2f} kg/m³, "
                f"Blast Furnace Slag: {row['blast_furnace_slag']:.2f} kg/m³, "
                f"Fly Ash: {row['fly_ash']:.2f} kg/m³, "
                f"Water: {row['water']:.2f} kg/m³, "
                f"Superplasticizer: {row['superplasticizer']:.2f} kg/m³, "
                f"Coarse Aggregate: {row['coarse_aggregate']:.2f} kg/m³, "
                f"Fine Aggregate: {row['fine_aggregate']:.2f} kg/m³, "
                f"Age: {row['age']} days"
            )
            
            response = llm_model.predict(instruction, input_text)
            predicted_value = self.extract_strength_from_text(response)
            
            if predicted_value is not None:
                predictions.append(predicted_value)
                valid_indices.append(idx)
        
        # Filter y_test to match valid predictions
        y_test_filtered = y_test.loc[valid_indices]
        predictions = np.array(predictions)
        
        if len(predictions) == 0:
            print("Warning: No valid predictions extracted from LLM")
            return {
                'test_rmse': float('inf'),
                'test_mae': float('inf'),
                'test_r2': -float('inf'),
                'predictions': np.array([]),
                'valid_ratio': 0.0
            }
        
        rmse = np.sqrt(mean_squared_error(y_test_filtered, predictions))
        mae = mean_absolute_error(y_test_filtered, predictions)
        r2 = r2_score(y_test_filtered, predictions)
        valid_ratio = len(predictions) / len(X_test)
        
        print(f"LLM Model:")
        print(f"  Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        print(f"  Valid predictions: {len(predictions)}/{len(X_test)} ({valid_ratio:.1%})")
        
        return {
            'test_rmse': rmse,
            'test_mae': mae,
            'test_r2': r2,
            'predictions': predictions,
            'valid_ratio': valid_ratio
        }
    
    def compare_models(
        self,
        ml_results: Dict[str, Dict[str, float]],
        llm_results: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Create comparison table of all models.
        
        Args:
            ml_results: Results from traditional ML models
            llm_results: Results from LLM model
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        # Add ML models
        for model_name, metrics in ml_results.items():
            comparison_data.append({
                'Model': model_name,
                'Type': 'Traditional ML',
                'RMSE': metrics['test_rmse'],
                'MAE': metrics['test_mae'],
                'R²': metrics['test_r2']
            })
        
        # Add LLM
        comparison_data.append({
            'Model': 'LLM (Fine-tuned)',
            'Type': 'LLM',
            'RMSE': llm_results['test_rmse'],
            'MAE': llm_results['test_mae'],
            'R²': llm_results['test_r2']
        })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('RMSE')
        
        return df
    
    def plot_comparison(
        self,
        comparison_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Create visualization comparing model performance.
        
        Args:
            comparison_df: DataFrame with comparison metrics
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RMSE comparison
        axes[0].barh(comparison_df['Model'], comparison_df['RMSE'])
        axes[0].set_xlabel('RMSE')
        axes[0].set_title('Root Mean Squared Error (Lower is Better)')
        axes[0].invert_yaxis()
        
        # MAE comparison
        axes[1].barh(comparison_df['Model'], comparison_df['MAE'])
        axes[1].set_xlabel('MAE')
        axes[1].set_title('Mean Absolute Error (Lower is Better)')
        axes[1].invert_yaxis()
        
        # R² comparison
        axes[2].barh(comparison_df['Model'], comparison_df['R²'])
        axes[2].set_xlabel('R²')
        axes[2].set_title('R² Score (Higher is Better)')
        axes[2].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.close()
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ):
        """
        Plot predicted vs actual values for multiple models.
        
        Args:
            y_true: True values
            predictions_dict: Dictionary mapping model names to predictions
            save_path: Optional path to save the figure
        """
        n_models = len(predictions_dict)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_models > 1 else axes
        
        for idx, (model_name, predictions) in enumerate(predictions_dict.items()):
            ax = axes[idx] if n_models > 1 else axes[0]
            
            # Ensure predictions and y_true have same length
            min_len = min(len(predictions), len(y_true))
            y_plot = y_true[:min_len]
            pred_plot = predictions[:min_len]
            
            ax.scatter(y_plot, pred_plot, alpha=0.5)
            
            # Add perfect prediction line
            min_val = min(y_plot.min(), pred_plot.min())
            max_val = max(y_plot.max(), pred_plot.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_xlabel('Actual Strength (MPa)')
            ax.set_ylabel('Predicted Strength (MPa)')
            ax.set_title(f'{model_name}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions plot saved to {save_path}")
        
        plt.close()


def compare_all_models(
    ml_results: Dict[str, Dict[str, float]],
    llm_results: Dict[str, float],
    y_test: np.ndarray,
    output_dir: str = "./results"
) -> pd.DataFrame:
    """
    Convenience function to compare all models and generate visualizations.
    
    Args:
        ml_results: Results from traditional ML models
        llm_results: Results from LLM
        y_test: True test values
        output_dir: Directory to save results
        
    Returns:
        Comparison DataFrame
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    comparator = ModelComparison()
    
    # Create comparison table
    comparison_df = comparator.compare_models(ml_results, llm_results)
    
    # Save comparison table
    comparison_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    print(f"\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Plot comparison
    comparator.plot_comparison(
        comparison_df,
        save_path=f"{output_dir}/comparison_metrics.png"
    )
    
    # Plot predictions
    predictions_dict = {
        name: results['predictions'] 
        for name, results in ml_results.items()
    }
    predictions_dict['LLM'] = llm_results['predictions']
    
    comparator.plot_predictions(
        y_test,
        predictions_dict,
        save_path=f"{output_dir}/predictions_vs_actual.png"
    )
    
    return comparison_df
