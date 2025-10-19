"""
Data loading and preprocessing utilities for concrete mix design.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split


class ConcreteDataset:
    """
    Handler for concrete mix design dataset.
    
    Expected columns:
    - Input features: cement, blast_furnace_slag, fly_ash, water, 
                     superplasticizer, coarse_aggregate, fine_aggregate, age
    - Target: compressive_strength (MPa)
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the concrete dataset handler.
        
        Args:
            data_path: Path to the CSV file containing concrete data.
                      If None, generates synthetic data for demonstration.
        """
        self.data_path = data_path
        self.data = None
        self.feature_columns = [
            'cement', 'blast_furnace_slag', 'fly_ash', 'water',
            'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age'
        ]
        self.target_columns = ['compressive_strength']
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV or generate synthetic data."""
        if self.data_path:
            self.data = pd.read_csv(self.data_path)
        else:
            # Generate synthetic data for demonstration
            self.data = self._generate_synthetic_data()
        return self.data
    
    def _generate_synthetic_data(self, n_samples: int = 1030) -> pd.DataFrame:
        """
        Generate synthetic concrete data based on typical ranges.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic concrete data
        """
        np.random.seed(42)
        
        data = {
            'cement': np.random.uniform(100, 540, n_samples),
            'blast_furnace_slag': np.random.uniform(0, 360, n_samples),
            'fly_ash': np.random.uniform(0, 200, n_samples),
            'water': np.random.uniform(120, 250, n_samples),
            'superplasticizer': np.random.uniform(0, 32, n_samples),
            'coarse_aggregate': np.random.uniform(800, 1145, n_samples),
            'fine_aggregate': np.random.uniform(594, 992, n_samples),
            'age': np.random.choice([1, 3, 7, 14, 28, 56, 91, 180, 365], n_samples),
        }
        
        # Generate compressive strength with a simplified formula
        df = pd.DataFrame(data)
        df['compressive_strength'] = (
            0.2 * df['cement'] +
            0.15 * df['blast_furnace_slag'] +
            0.1 * df['fly_ash'] -
            0.3 * df['water'] +
            2.5 * df['superplasticizer'] +
            0.01 * df['coarse_aggregate'] +
            0.01 * df['fine_aggregate'] +
            0.15 * df['age'] +
            np.random.normal(0, 5, n_samples)
        )
        
        # Ensure realistic range
        df['compressive_strength'] = df['compressive_strength'].clip(2.33, 82.6)
        
        return df
    
    def prepare_train_test_split(
        self, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            self.load_data()
            
        X = self.data[self.feature_columns]
        y = self.data[self.target_columns[0]]
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def prepare_llm_dataset(self) -> list:
        """
        Convert concrete data to instruction-following format for LLM fine-tuning.
        
        Returns:
            List of dictionaries with 'instruction', 'input', 'output' keys
        """
        if self.data is None:
            self.load_data()
        
        llm_data = []
        
        for idx, row in self.data.iterrows():
            instruction = (
                "Given the following concrete mix design parameters, "
                "predict the compressive strength in MPa."
            )
            
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
            
            output_text = f"The predicted compressive strength is {row['compressive_strength']:.2f} MPa."
            
            llm_data.append({
                'instruction': instruction,
                'input': input_text,
                'output': output_text
            })
        
        return llm_data


def load_concrete_data(data_path: Optional[str] = None) -> ConcreteDataset:
    """
    Convenience function to load concrete dataset.
    
    Args:
        data_path: Path to CSV file or None for synthetic data
        
    Returns:
        ConcreteDataset instance with loaded data
    """
    dataset = ConcreteDataset(data_path)
    dataset.load_data()
    return dataset
