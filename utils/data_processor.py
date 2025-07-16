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