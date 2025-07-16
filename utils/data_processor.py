"""
Data processing module for Water Potability Analysis System
"""
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.impute import SimpleImputer
from typing import Dict, Any, Optional, List, Tuple
import logging

from core.base_classes import IDataProcessor, Subject, DataValidator
from config.settings import Settings


class DataProcessor(IDataProcessor, Subject):
    """Enhanced data processor with comprehensive analysis"""
    
    def __init__(self):
        super().__init__()
        self.df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.imputer = SimpleImputer(strategy=Settings.DATA.imputation_strategy)
        self.logger = logging.getLogger(__name__)
    
    @st.cache_data
    def load_data(_self) -> Optional[pd.DataFrame]:
        """Load and cache the water potability dataset"""
        try:
            _self.logger.info("Loading data from URL")
            df = pd.read_csv(Settings.DATA.url)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            _self.df = df
            _self.notify("data_loaded", {"shape": df.shape, "columns": list(df.columns)})
            
            return df
        except Exception as e:
            _self.logger.error(f"Error loading data: {e}")
            st.error(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data with comprehensive cleaning"""
        try:
            self.logger.info("Starting data preprocessing")
            processed_df = df.copy()
            
            # Handle missing values
            numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
            processed_df[numeric_columns] = self.imputer.fit_transform(processed_df[numeric_columns])
            
            # Remove duplicates
            initial_shape = processed_df.shape
            processed_df = processed_df.drop_duplicates()
            duplicates_removed = initial_shape[0] - processed_df.shape[0]
            
            self.processed_df = processed_df
            self.notify("data_preprocessed", {
                "duplicates_removed": duplicates_removed,
                "final_shape": processed_df.shape
            })
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {e}")
            raise
    
    def prepare_features_target(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variables for ML models"""
        if df is None:
            df = self.processed_df or self.df
        
        if df is None:
            raise ValueError("No data available for feature preparation")
        
        X = df[Settings.PARAMETERS].copy()
        y = df['Potability'].copy()
        
        return X, y
    
    # Add to existing data_processor.py

class StatisticalAnalyzer:
    """Analyzer for statistical metrics"""
    
    def __init__(self):
        self.results = {}
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis"""
        results = {
            'descriptive_stats': self._calculate_descriptive_stats(data),
            'correlation_analysis': self._calculate_correlations(data),
            'class_balance': self._analyze_class_balance(data)
        }
        self.results = results
        return results
    
    def _calculate_descriptive_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate descriptive statistics"""
        stats = {}
        for param in Settings.PARAMETERS:
            if param in df.columns:
                stats[param] = {
                    'mean': df[param].mean(),
                    'median': df[param].median(),
                    'std': df[param].std(),
                    'min': df[param].min(),
                    'max': df[param].max(),
                    'skewness': df[param].skew(),
                    'kurtosis': df[param].kurtosis()
                }
        return stats
    
    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlations with target variable"""
        correlations = {}
        if 'Potability' in df.columns:
            for param in Settings.PARAMETERS:
                if param in df.columns:
                    correlations[param] = df[param].corr(df['Potability'])
        return correlations
    
    def _analyze_class_balance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze class balance"""
        if 'Potability' not in df.columns:
            return {}
        
        class_counts = df['Potability'].value_counts()
        total_samples = len(df)
        
        return {
            'potable_count': class_counts.get(1, 0),
            'non_potable_count': class_counts.get(0, 0),
            'potable_percentage': (class_counts.get(1, 0) / total_samples) * 100,
            'balance_ratio': class_counts.get(1, 0) / class_counts.get(0, 1) if class_counts.get(0, 0) > 0 else 0
        }


# Add to DataProcessor class
def calculate_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Calculate comprehensive statistics"""
    if df is None:
        df = self.df
    
    if df is None:
        return {}
    
    try:
        analyzer = StatisticalAnalyzer()
        statistical_results = analyzer.analyze(df)
        
        combined_stats = {
            'basic_info': {
                'total_samples': len(df),
                'total_features': len(df.columns),
            },
            'statistical_analysis': statistical_results
        }
        
        self.notify("statistics_calculated", combined_stats)
        return combined_stats
        
    except Exception as e:
        self.logger.error(f"Error calculating statistics: {e}")
        return {}

def get_correlation_matrix(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Get correlation matrix for parameters"""
    if df is None:
        df = self.df
    
    if df is None:
        return pd.DataFrame()
    
    return df[Settings.PARAMETERS + ['Potability']].corr()