"""
Data loading and preprocessing utilities for concrete mix design.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from sklearn.model_selection import train_test_split
import os


class ConcreteDataset:
    """
    Handler for concrete mix design dataset with multi-task support.
    
    Expected columns for compression data:
    - Input features: w (water), c (cement), sf, fa, sp (superplasticizer), sin_fa, ex_cl, od, age
    - Target: fcr (compressive strength)
    
    Expected columns for tensile data:
    - Input features: Agua (water), Agregados (aggregates), Cemento (cement), Factor
    - Target: Resistencia a Tracción (tensile strength)
    """
    
    def __init__(self, compression_path: Optional[str] = None, tensile_path: Optional[str] = None):
        """
        Initialize the concrete dataset handler.
        
        Args:
            compression_path: Path to the compression CSV file
            tensile_path: Path to the tensile CSV file
        """
        self.compression_path = compression_path
        self.tensile_path = tensile_path
        self.data = None
        self.compression_data = None
        self.tensile_data = None
        
        # Unified feature columns (mapped from both datasets)
        self.feature_columns = [
            'water', 'cement', 'aggregates', 'superplasticizer', 'age', 'factor'
        ]
        self.target_columns = ['compressive_strength', 'tensile_strength']
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV files or generate synthetic data."""
        if self.compression_path and self.tensile_path:
            self._load_multitask_data()
        elif self.compression_path:
            self._load_compression_data()
        elif self.tensile_path:
            self._load_tensile_data()
        else:
            # Generate synthetic data for demonstration
            self.data = self._generate_synthetic_data()
        return self.data
    
    def _clean_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean numeric data by converting string numbers to float.
        
        Args:
            df: DataFrame with potentially string numeric values
            
        Returns:
            Cleaned DataFrame with proper numeric types
        """
        df_clean = df.copy()
        
        # Define numeric columns that need cleaning
        numeric_columns = ['w', 'c', 'sf', 'fa', 'sp', 'sin_fa', 'ex_cl', 'od', 'age', 'fcr',
                          'Agua (kg)', 'Agregados (kg)', 'Cemento (kg)', 'Factor (kg/m³)', 
                          'Resistencia a Tracción (MPa)']
        
        for col in numeric_columns:
            if col in df_clean.columns:
                # Convert to string first, then clean
                df_clean[col] = df_clean[col].astype(str)
                
                # Remove commas and other non-numeric characters except decimal points
                df_clean[col] = df_clean[col].str.replace(',', '', regex=False)
                df_clean[col] = df_clean[col].str.replace(' ', '', regex=False)
                
                # Convert to numeric, errors='coerce' will turn invalid values to NaN
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        return df_clean
    
    def _load_compression_data(self):
        """Load compression data and map to unified schema."""
        df = pd.read_csv(self.compression_path, encoding='utf-8')
        df = self._clean_numeric_data(df)
        
        # Map compression columns to unified schema
        compression_mapped = pd.DataFrame({
            'water': df['w'],
            'cement': df['c'],
            'aggregates': df['fa'] + df['sin_fa'] + df['ex_cl'] + df['od'],  # Combined aggregates
            'superplasticizer': df['sp'],
            'age': df['age'],
            'factor': np.nan,  # Not available in compression data
            'compressive_strength': df['fcr'],
            'tensile_strength': np.nan,  # Not available
            'task': 'compression'
        })
        
        self.compression_data = compression_mapped
        self.data = compression_mapped
    
    def _load_tensile_data(self):
        """Load tensile data and map to unified schema."""
        try:
            df = pd.read_csv(self.tensile_path, encoding='utf-8')
        except UnicodeDecodeError:
            # Try with different encodings
            try:
                df = pd.read_csv(self.tensile_path, encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv(self.tensile_path, encoding='cp1252')
        
        df = self._clean_numeric_data(df)
        
        # Map tensile columns to unified schema
        tensile_mapped = pd.DataFrame({
            'water': df['Agua (kg)'],
            'cement': df['Cemento (kg)'],
            'aggregates': df['Agregados (kg)'],
            'superplasticizer': np.nan,  # Not available in tensile data
            'age': 28,  # Assume 28 days (not specified in tensile data)
            'factor': df['Factor (kg/m³)'],
            'compressive_strength': np.nan,  # Not available
            'tensile_strength': df['Resistencia a Tracción (MPa)'],
            'task': 'tensile'
        })
        
        self.tensile_data = tensile_mapped
        self.data = tensile_mapped
    
    def _load_multitask_data(self):
        """Load both compression and tensile data, combine into unified dataset."""
        # Load compression data
        comp_df = pd.read_csv(self.compression_path, encoding='utf-8')
        comp_df = self._clean_numeric_data(comp_df)
        compression_mapped = pd.DataFrame({
            'water': comp_df['w'],
            'cement': comp_df['c'],
            'aggregates': comp_df['fa'] + comp_df['sin_fa'] + comp_df['ex_cl'] + comp_df['od'],
            'superplasticizer': comp_df['sp'],
            'age': comp_df['age'],
            'factor': np.nan,
            'compressive_strength': comp_df['fcr'],
            'tensile_strength': np.nan,
            'task': 'compression'
        })
        
        # Load tensile data with encoding handling
        try:
            tens_df = pd.read_csv(self.tensile_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                tens_df = pd.read_csv(self.tensile_path, encoding='latin-1')
            except UnicodeDecodeError:
                tens_df = pd.read_csv(self.tensile_path, encoding='cp1252')
        
        tens_df = self._clean_numeric_data(tens_df)
        
        tensile_mapped = pd.DataFrame({
            'water': tens_df['Agua (kg)'],
            'cement': tens_df['Cemento (kg)'],
            'aggregates': tens_df['Agregados (kg)'],
            'superplasticizer': np.nan,
            'age': 28,
            'factor': tens_df['Factor (kg/m³)'],
            'compressive_strength': np.nan,
            'tensile_strength': tens_df['Resistencia a Tracción (MPa)'],
            'task': 'tensile'
        })
        
        # Combine datasets
        self.compression_data = compression_mapped
        self.tensile_data = tensile_mapped
        self.data = pd.concat([compression_mapped, tensile_mapped], ignore_index=True)
        
        print(f"Loaded {len(compression_mapped)} compression samples")
        print(f"Loaded {len(tensile_mapped)} tensile samples")
        print(f"Total samples: {len(self.data)}")
    
    def _generate_synthetic_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """
        Generate synthetic concrete data based on concrete engineering principles.
        
        Args:
            n_samples: Number of synthetic samples to generate
            
        Returns:
            DataFrame with synthetic concrete data
        """
        np.random.seed(42)
        
        # Generate realistic concrete mix proportions
        data = {
            'water': np.random.uniform(120, 250, n_samples),
            'cement': np.random.uniform(200, 600, n_samples),
            'aggregates': np.random.uniform(800, 2000, n_samples),
            'superplasticizer': np.random.uniform(0, 32, n_samples),
            'age': np.random.choice([1, 3, 7, 14, 28, 56, 91, 180, 365], n_samples),
            'factor': np.random.uniform(1.0, 1.2, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # More sophisticated compressive strength formula based on concrete engineering
        # Based on Abrams' law and ACI 318 relationships
        w_c_ratio = df['water'] / df['cement']  # Water-cement ratio
        
        # Base strength from water-cement ratio (Abrams' law)
        base_strength = 3000 / (w_c_ratio ** 2.5)  # ACI formula
        
        # Age factor (strength development over time)
        age_factor = np.where(df['age'] >= 28, 
                             1.0 + 0.1 * np.log(df['age'] / 28),  # Logarithmic growth after 28 days
                             0.4 + 0.6 * (df['age'] / 28))  # Linear growth up to 28 days
        
        # Superplasticizer effect (increases workability, can increase strength)
        sp_factor = 1.0 + 0.05 * (df['superplasticizer'] / 32)  # Up to 5% increase
        
        # Aggregate effect (more aggregates generally increase strength)
        agg_factor = 1.0 + 0.02 * ((df['aggregates'] - 1000) / 1000)  # Normalized effect
        
        # Add realistic noise
        noise = np.random.normal(0, 0.1, n_samples)  # 10% coefficient of variation
        
        df['compressive_strength'] = base_strength * age_factor * sp_factor * agg_factor * (1 + noise)
        
        # Generate tensile strength using more accurate relationship
        # Tensile strength is typically 8-12% of compressive strength
        tensile_ratio = np.random.uniform(0.08, 0.12, n_samples)
        df['tensile_strength'] = df['compressive_strength'] * tensile_ratio
        
        # Ensure realistic ranges based on actual concrete properties
        df['compressive_strength'] = df['compressive_strength'].clip(5.0, 100.0)  # 5-100 MPa
        df['tensile_strength'] = df['tensile_strength'].clip(0.5, 12.0)  # 0.5-12 MPa
        
        # Add some correlation between compression and tensile data
        # Some samples should have both properties
        both_properties = np.random.choice([True, False], n_samples, p=[0.7, 0.3])
        
        # Create task indicators
        df['task'] = 'synthetic'
        
        # Add some compression-only and tensile-only samples for realism
        compression_only = np.random.choice([True, False], n_samples, p=[0.2, 0.8])
        tensile_only = np.random.choice([True, False], n_samples, p=[0.1, 0.9])
        
        # Set some tensile strengths to NaN for compression-only samples
        df.loc[compression_only, 'tensile_strength'] = np.nan
        df.loc[tensile_only, 'compressive_strength'] = np.nan
        
        print(f"Generated {n_samples} synthetic samples:")
        print(f"  - Both properties: {both_properties.sum()}")
        print(f"  - Compression only: {compression_only.sum()}")
        print(f"  - Tensile only: {tensile_only.sum()}")
        
        return df
    
    def load_data_with_synthetic(self, synthetic_samples: int = 2000):
        """
        Load real data and combine with synthetic data for enhanced training.
        
        Args:
            synthetic_samples: Number of synthetic samples to generate
            
        Returns:
            Combined DataFrame with real and synthetic data
        """
        # Load real data first
        if self.compression_path and self.tensile_path:
            self._load_multitask_data()
            real_data = self.data.copy()
            print(f"Loaded {len(real_data)} real samples")
        else:
            real_data = pd.DataFrame()
            print("No real data available, using only synthetic data")
        
        # Generate synthetic data
        synthetic_data = self._generate_synthetic_data(synthetic_samples)
        
        # Combine real and synthetic data
        if len(real_data) > 0:
            self.data = pd.concat([real_data, synthetic_data], ignore_index=True)
            print(f"Combined dataset: {len(real_data)} real + {len(synthetic_data)} synthetic = {len(self.data)} total")
        else:
            self.data = synthetic_data
            print(f"Using {len(self.data)} synthetic samples")
        
        return self.data
    
    def _get_compression_instruction(self) -> str:
        """Get compression strength prediction instruction with few-shot examples."""
        return """You are a concrete engineering expert. Predict the compressive strength of concrete based on mix design parameters.

Examples:
Input: Water: 180.00 kg/m³, Cement: 350.00 kg/m³, Aggregates: 1200.00 kg/m³, Superplasticizer: 5.00 kg/m³, Age: 28 days
Output: 32.50 MPa

Input: Water: 200.00 kg/m³, Cement: 400.00 kg/m³, Aggregates: 1500.00 kg/m³, Superplasticizer: 0.00 kg/m³, Age: 28 days
Output: 28.75 MPa

Input: Water: 160.00 kg/m³, Cement: 450.00 kg/m³, Aggregates: 1800.00 kg/m³, Superplasticizer: 12.00 kg/m³, Age: 28 days
Output: 45.20 MPa

Now predict the compressive strength for the given mix design. Respond with only a number followed by 'MPa'."""
    
    def _get_tensile_instruction(self) -> str:
        """Get tensile strength prediction instruction with few-shot examples."""
        return """You are a concrete engineering expert. Predict the tensile strength of concrete based on mix design parameters.

Examples:
Input: Water: 180.00 kg/m³, Cement: 350.00 kg/m³, Aggregates: 1200.00 kg/m³, Age: 28 days, Factor: 1.10 kg/m³
Output: 3.25 MPa

Input: Water: 200.00 kg/m³, Cement: 400.00 kg/m³, Aggregates: 1500.00 kg/m³, Age: 28 days, Factor: 1.05 kg/m³
Output: 2.88 MPa

Input: Water: 160.00 kg/m³, Cement: 450.00 kg/m³, Aggregates: 1800.00 kg/m³, Age: 28 days, Factor: 1.15 kg/m³
Output: 4.52 MPa

Now predict the tensile strength for the given mix design. Respond with only a number followed by 'MPa'."""
    
    def prepare_train_val_test_split(
        self, 
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets with 70/15/15 split.
        Returns multi-task targets (both compression and tensile columns).
        
        Args:
            train_size: Proportion of data for training
            val_size: Proportion of data for validation
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
            where y_* are DataFrames with both compressive_strength and tensile_strength columns
        """
        if self.data is None:
            self.load_data()
        
        # Ensure splits sum to 1.0
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"
        
        X = self.data[self.feature_columns]
        y = self.data[['compressive_strength', 'tensile_strength']]  # Multi-task targets
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_size + test_size), random_state=random_state
        )
        
        # Second split: val vs test
        val_test_size = test_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_test_size, random_state=random_state
        )
        
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_single_task_split(self, task='compression', train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        """
        Prepare train/validation/test split for single task (compression or tensile only).
        This method filters out NaN values for the specific task.
        
        Args:
            task: 'compression' or 'tensile'
            train_size: Proportion of data for training
            val_size: Proportion of data for validation  
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if self.data is None:
            self.load_data()
        
        # Filter data for the specific task (remove NaN values)
        if task == 'compression':
            task_data = self.data.dropna(subset=['compressive_strength'])
            target_col = 'compressive_strength'
        elif task == 'tensile':
            task_data = self.data.dropna(subset=['tensile_strength'])
            target_col = 'tensile_strength'
        else:
            raise ValueError("Task must be 'compression' or 'tensile'")
        
        # Prepare features and target
        X = task_data[self.feature_columns].copy()
        y = task_data[[target_col]].copy()
        
        # Handle NaN values in features by filling with appropriate defaults
        for col in X.columns:
            if X[col].isna().any():
                if col == 'superplasticizer':
                    # Superplasticizer not used in tensile data, fill with 0
                    fill_val = 0.0
                elif col == 'factor':
                    # Factor not used in compression data, fill with 1.0 (typical factor)
                    fill_val = 1.0
                else:
                    # For other columns, use median if available, otherwise 0
                    fill_val = X[col].median()
                    if pd.isna(fill_val):
                        fill_val = 0.0
                
                X[col] = X[col].fillna(fill_val)
                print(f"  Filled {X[col].isna().sum()} NaN values in '{col}' with: {fill_val:.2f}")
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_size + test_size), random_state=random_state, stratify=None
        )
        
        # Second split: val vs test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(test_size / (val_size + test_size)), 
            random_state=random_state, stratify=None
        )
        
        print(f"{task.capitalize()} data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_train_test_split(
        self, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets (backward compatibility).
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            self.load_data()
            
        X = self.data[self.feature_columns]
        y = self.data[self.target_columns]
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def prepare_multitask_llm_dataset(self, augment_data: bool = True) -> List[Dict[str, str]]:
        """
        Convert concrete data to instruction-following format for multi-task LLM fine-tuning.
        
        Args:
            augment_data: Whether to augment the dataset with variations
            
        Returns:
            List of dictionaries with 'instruction', 'input', 'output', 'task' keys
        """
        if self.data is None:
            self.load_data()
        
        llm_data = []
        
        for idx, row in self.data.iterrows():
            # Determine task based on available targets
            if pd.notna(row['compressive_strength']) and pd.notna(row['tensile_strength']):
                # Both targets available - create two examples
                tasks = ['compression', 'tensile']
            elif pd.notna(row['compressive_strength']):
                tasks = ['compression']
            elif pd.notna(row['tensile_strength']):
                tasks = ['tensile']
            else:
                continue  # Skip if no targets available
            
            for task in tasks:
                if task == 'compression':
                    instruction = self._get_compression_instruction()
                    output_text = f"{row['compressive_strength']:.2f} MPa"
                else:  # tensile
                    instruction = self._get_tensile_instruction()
                    output_text = f"{row['tensile_strength']:.2f} MPa"
                
                # Create input text with available features
                input_parts = []
                if pd.notna(row['water']) and isinstance(row['water'], (int, float)):
                    input_parts.append(f"Water: {row['water']:.2f} kg/m³")
                if pd.notna(row['cement']) and isinstance(row['cement'], (int, float)):
                    input_parts.append(f"Cement: {row['cement']:.2f} kg/m³")
                if pd.notna(row['aggregates']) and isinstance(row['aggregates'], (int, float)):
                    input_parts.append(f"Aggregates: {row['aggregates']:.2f} kg/m³")
                if pd.notna(row['superplasticizer']) and isinstance(row['superplasticizer'], (int, float)):
                    input_parts.append(f"Superplasticizer: {row['superplasticizer']:.2f} kg/m³")
                if pd.notna(row['age']) and isinstance(row['age'], (int, float)):
                    input_parts.append(f"Age: {int(row['age'])} days")
                if pd.notna(row['factor']) and isinstance(row['factor'], (int, float)):
                    input_parts.append(f"Factor: {row['factor']:.2f} kg/m³")
                
                input_text = ", ".join(input_parts)
                
                llm_data.append({
                    'instruction': instruction,
                    'input': input_text,
                    'output': output_text,
                    'task': task
                })
        
        # Data augmentation: create variations of instructions
        if augment_data and len(llm_data) > 0:
            augmented_data = []
            for item in llm_data:
                # Original
                augmented_data.append(item)
                
                # Variation 1: Different instruction style
                if item['task'] == 'compression':
                    var1_instruction = (
                        "Calculate the compressive strength of concrete with these mix parameters. "
                        "Provide the result in MPa."
                    )
                else:
                    var1_instruction = (
                        "Calculate the tensile strength of concrete with these mix parameters. "
                        "Provide the result in MPa."
                    )
                
                augmented_data.append({
                    'instruction': var1_instruction,
                    'input': item['input'],
                    'output': item['output'],
                    'task': item['task']
                })
                
                # Variation 2: More technical instruction
                if item['task'] == 'compression':
                    var2_instruction = (
                        "As a structural engineer, determine the 28-day compressive strength "
                        "for this concrete mix design. Answer in MPa."
                    )
                else:
                    var2_instruction = (
                        "As a structural engineer, determine the tensile strength "
                        "for this concrete mix design. Answer in MPa."
                    )
                
                augmented_data.append({
                    'instruction': var2_instruction,
                    'input': item['input'],
                    'output': item['output'],
                    'task': item['task']
                })
            
            llm_data = augmented_data
        
        print(f"Generated {len(llm_data)} LLM training samples")
        return llm_data
    
    def prepare_llm_dataset(self) -> List[Dict[str, str]]:
        """
        Convert concrete data to instruction-following format for LLM fine-tuning.
        Backward compatibility method.
        
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
                f"Water: {row['water']:.2f} kg/m³, "
                f"Cement: {row['cement']:.2f} kg/m³, "
                f"Aggregates: {row['aggregates']:.2f} kg/m³, "
                f"Superplasticizer: {row['superplasticizer']:.2f} kg/m³, "
                f"Age: {row['age']} days"
            )
            
            if pd.notna(row['compressive_strength']):
                output_text = f"The predicted compressive strength is {row['compressive_strength']:.2f} MPa."
            else:
                output_text = f"The predicted tensile strength is {row['tensile_strength']:.2f} MPa."
            
            llm_data.append({
                'instruction': instruction,
                'input': input_text,
                'output': output_text
            })
        
        return llm_data


def load_concrete_data(
    compression_path: Optional[str] = None, 
    tensile_path: Optional[str] = None
) -> ConcreteDataset:
    """
    Convenience function to load concrete dataset.
    
    Args:
        compression_path: Path to compression CSV file
        tensile_path: Path to tensile CSV file
        
    Returns:
        ConcreteDataset instance with loaded data
    """
    dataset = ConcreteDataset(compression_path, tensile_path)
    dataset.load_data()
    return dataset
