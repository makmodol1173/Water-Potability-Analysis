"""
Configuration settings for the Water Potability Analysis System
"""
from dataclasses import dataclass
from typing import Dict, List, Any
from enum import Enum


class ModelType(Enum):
    """Enumeration for available model types"""
    RANDOM_FOREST = "Random Forest"
    LOGISTIC_REGRESSION = "Logistic Regression"
    SVM = "Support Vector Machine"
    GRADIENT_BOOSTING = "Gradient Boosting"


@dataclass
class AppConfiguration:
    """Application configuration dataclass"""
    page_title: str = "Water Potability Analysis System"
    page_icon: str = "💧"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"


@dataclass
class DataConfiguration:
    """Data configuration dataclass"""
    url: str = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/water_potability_preprocessed-aP2VS7drsoWULn1qmITGHQDpRcDEhe.csv"
    imputation_strategy: str = 'median'
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfiguration:
    """Model configuration dataclass"""
    cv_folds: int = 5
    random_state: int = 42
    rf_estimators: int = 100
    lr_max_iter: int = 1000


class Settings:
    """Centralized settings management class"""
    
    APP = AppConfiguration()
    DATA = DataConfiguration()
    MODEL = ModelConfiguration()
    
    PARAMETERS: List[str] = [
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ]
    
    COLORS: Dict[str, str] = {
        'potable': '#28a745',
        'non_potable': '#dc3545',
        'primary': '#1f77b4'
    }
    
    @classmethod
    def get_parameter_display_name(cls, param: str) -> str:
        """Get display name for parameter"""
        return param.replace('_', ' ').title()