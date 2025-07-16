"""
Enhanced data processing module with advanced OOP concepts
"""
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, Any, Optional, List, Tuple
import logging

from core.base_classes import IDataProcessor, BaseAnalyzer, Subject, DataValidator
from config.settings import Settings


class DataQualityAnalyzer(BaseAnalyzer):
    """Analyzer for data quality metrics"""
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality"""
        results = {
            'missing_values': self._analyze_missing_values(data),
            'outliers': self._analyze_outliers(data),
            'duplicates': self._analyze_duplicates(data),
            'data_types': self._analyze_data_types(data)
        }
        self._results = results
        return results
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values"""
        missing_count = df.isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        
        return {
            'total_missing': missing_count.sum(),
            'missing_by_column': missing_count.to_dict(),
            'missing_percentage_by_column': missing_percentage.to_dict(),
            'overall_completeness': ((df.size - missing_count.sum()) / df.size) * 100
        }
    
    def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers using IQR method"""
        outlier_info = {}
        
        for column in Settings.PARAMETERS:
            if column in df.columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                outlier_info[column] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        return outlier_info
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate records"""
        return {
            'duplicate_count': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
            'unique_records': len(df.drop_duplicates())
        }
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze data types"""
        return df.dtypes.astype(str).to_dict()


class StatisticalAnalyzer(BaseAnalyzer):
    """Analyzer for statistical metrics"""
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis"""
        results = {
            'descriptive_stats': self._calculate_descriptive_stats(data),
            'correlation_analysis': self._calculate_correlations(data),
            'distribution_analysis': self._analyze_distributions(data),
            'class_balance': self._analyze_class_balance(data)
        }
        self._results = results
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
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze parameter distributions"""
        distributions = {}
        for param in Settings.PARAMETERS:
            if param in df.columns:
                distributions[param] = {
                    'q25': df[param].quantile(0.25),
                    'q50': df[param].quantile(0.50),
                    'q75': df[param].quantile(0.75),
                    'iqr': df[param].quantile(0.75) - df[param].quantile(0.25)
                }
        return distributions
    
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
            'non_potable_percentage': (class_counts.get(0, 0) / total_samples) * 100,
            'balance_ratio': class_counts.get(1, 0) / class_counts.get(0, 1) if class_counts.get(0, 0) > 0 else 0
        }


class DataProcessor(IDataProcessor, Subject):
    """Enhanced data processor with observer pattern and comprehensive analysis"""
    
    def __init__(self):
        super().__init__()
        self.df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.imputer = SimpleImputer(strategy=Settings.DATA.imputation_strategy)
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # Initialize analyzers
        self.quality_analyzer = DataQualityAnalyzer("Data Quality")
        self.statistical_analyzer = StatisticalAnalyzer("Statistical Analysis")
        
        # Setup logging
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
            
            # Handle outliers (optional - can be enabled/disabled)
            processed_df = self._handle_outliers(processed_df)
            
            self.processed_df = processed_df
            self.notify("data_preprocessed", {
                "duplicates_removed": duplicates_removed,
                "final_shape": processed_df.shape
            })
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {e}")
            raise
    
    def _handle_outliers(self, df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
        """Handle outliers using specified method"""
        if method == "iqr":
            return self._remove_outliers_iqr(df)
        elif method == "zscore":
            return self._remove_outliers_zscore(df)
        else:
            return df
    
    def _remove_outliers_iqr(self, df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        cleaned_df = df.copy()
        
        for column in Settings.PARAMETERS:
            if column in cleaned_df.columns:
                Q1 = cleaned_df[column].quantile(0.25)
                Q3 = cleaned_df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                # Remove outliers
                cleaned_df = cleaned_df[
                    (cleaned_df[column] >= lower_bound) & 
                    (cleaned_df[column] <= upper_bound)
                ]
        
        return cleaned_df
    
    def _remove_outliers_zscore(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using Z-score method"""
        from scipy import stats
        
        cleaned_df = df.copy()
        
        for column in Settings.PARAMETERS:
            if column in cleaned_df.columns:
                z_scores = np.abs(stats.zscore(cleaned_df[column]))
                cleaned_df = cleaned_df[z_scores < threshold]
        
        return cleaned_df
    
    def calculate_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        if df is None:
            df = self.df
        
        if df is None:
            return {}
        
        try:
            # Run analyzers
            quality_results = self.quality_analyzer.analyze(df)
            statistical_results = self.statistical_analyzer.analyze(df)
            
            # Combine results
            combined_stats = {
                'basic_info': {
                    'total_samples': len(df),
                    'total_features': len(df.columns),
                    'memory_usage': df.memory_usage(deep=True).sum()
                },
                'data_quality': quality_results,
                'statistical_analysis': statistical_results
            }
            
            self.notify("statistics_calculated", combined_stats)
            return combined_stats
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            return {}
    
    def get_parameter_analysis(self, input_data: Dict[str, float]) -> pd.DataFrame:
        """Analyze input parameters against optimal ranges with validation"""
        # Validate input data
        is_valid, errors = DataValidator.validate_water_parameters(input_data)
        
        analysis_data = []
        
        for param, value in input_data.items():
            if param in Settings.PARAMETER_INFO:
                info = Settings.PARAMETER_INFO[param]
                
                # Check if value is within optimal range
                if info.optimal_min <= value <= info.optimal_max:
                    status = "✅ Optimal"
                    status_color = "success"
                elif info.range_min <= value <= info.range_max:
                    status = "⚠️ Acceptable"
                    status_color = "warning"
                else:
                    status = "❌ Outside range"
                    status_color = "danger"
                
                analysis_data.append({
                    'Parameter': Settings.get_parameter_display_name(param),
                    'Value': f"{value:.2f} {info.unit}",
                    'Optimal Range': f"{info.optimal_str} {info.unit}",
                    'Status': status,
                    'Description': info.description,
                    'Status_Color': status_color
                })
        
        df = pd.DataFrame(analysis_data)
        
        # Add validation errors if any
        if not is_valid:
            df['Validation_Errors'] = errors
        
        return df
    
    def prepare_features_target(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variables for ML models"""
        if df is None:
            df = self.processed_df or self.df
        
        if df is None:
            raise ValueError("No data available for feature preparation")
        
        X = df[Settings.PARAMETERS].copy()
        y = df['Potability'].copy()
        
        return X, y
    
    def get_correlation_matrix(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get correlation matrix for parameters"""
        if df is None:
            df = self.df
        
        if df is None:
            return pd.DataFrame()
        
        return df[Settings.PARAMETERS + ['Potability']].corr()
    
    def get_feature_engineering_suggestions(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        """Get suggestions for feature engineering"""
        if df is None:
            df = self.df
        
        if df is None:
            return []
        
        suggestions = []
        
        # Check for highly correlated features
        corr_matrix = self.get_correlation_matrix(df)
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        if high_corr_pairs:
            suggestions.append(f"Consider removing one feature from highly correlated pairs: {high_corr_pairs}")
        
        # Check for skewed distributions
        skewed_features = []
        stats = self.statistical_analyzer.get_results()
        if 'descriptive_stats' in stats:
            for param, param_stats in stats['descriptive_stats'].items():
                if abs(param_stats.get('skewness', 0)) > 1:
                    skewed_features.append(param)
        
        if skewed_features:
            suggestions.append(f"Consider log transformation for skewed features: {skewed_features}")
        
        # Suggest interaction features
        suggestions.append("Consider creating interaction features between pH and other chemical parameters")
        suggestions.append("Consider creating ratio features (e.g., Chloramines/Sulfate)")
        
        return suggestions
    
    def export_processed_data(self, filename: str = "processed_water_data.csv") -> bool:
        """Export processed data to CSV"""
        try:
            if self.processed_df is not None:
                self.processed_df.to_csv(filename, index=False)
                self.notify("data_exported", {"filename": filename, "shape": self.processed_df.shape})
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return False
