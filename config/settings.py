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


class VisualizationType(Enum):
    """Enumeration for visualization types"""
    PIE_CHART = "pie_chart"
    BAR_CHART = "bar_chart"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    RADAR_CHART = "radar_chart"
    THREE_D_SCATTER = "3d_scatter"


@dataclass
class AppConfiguration:
    """Application configuration dataclass"""
    page_title: str = "Water Potability Analysis System"
    page_icon: str = "💧"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    theme: str = "light"
    debug_mode: bool = False


@dataclass
class DataConfiguration:
    """Data configuration dataclass"""
    url: str = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/water_potability_preprocessed-aP2VS7drsoWULn1qmITGHQDpRcDEhe.csv"
    imputation_strategy: str = 'median'
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    sample_size_for_viz: int = 1000


@dataclass
class ModelConfiguration:
    """Model configuration dataclass"""
    cv_folds: int = 5
    random_state: int = 42
    rf_estimators: int = 100
    lr_max_iter: int = 1000
    svm_kernel: str = 'rbf'
    gb_estimators: int = 100
    gb_learning_rate: float = 0.1


@dataclass
class ParameterInfo:
    """Parameter information dataclass"""
    range_min: float
    range_max: float
    optimal_min: float
    optimal_max: float
    unit: str
    description: str
    
    @property
    def range_str(self) -> str:
        return f"{self.range_min}-{self.range_max}"
    
    @property
    def optimal_str(self) -> str:
        return f"{self.optimal_min}-{self.optimal_max}"


class Settings:
    """Centralized settings management class"""
    
    # Application Configuration
    APP = AppConfiguration()
    
    # Data Configuration
    DATA = DataConfiguration()
    
    # Model Configuration
    MODEL = ModelConfiguration()
    
    # Water Quality Parameters
    PARAMETERS: List[str] = [
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ]
    
    # Parameter Information
    PARAMETER_INFO: Dict[str, ParameterInfo] = {
        'ph': ParameterInfo(0, 14, 6.5, 8.5, '', 'Acidity/alkalinity level'),
        'Hardness': ParameterInfo(47, 323, 60, 120, 'mg/L', 'Mineral content'),
        'Solids': ParameterInfo(320, 61227, 0, 500, 'ppm', 'Total dissolved solids'),
        'Chloramines': ParameterInfo(0.35, 13.1, 0, 4, 'ppm', 'Disinfectant levels'),
        'Sulfate': ParameterInfo(129, 481, 0, 250, 'mg/L', 'Sulfate concentration'),
        'Conductivity': ParameterInfo(181, 753, 200, 800, 'μS/cm', 'Electrical conductivity'),
        'Organic_carbon': ParameterInfo(2.2, 28.3, 0, 2, 'ppm', 'Organic carbon content'),
        'Trihalomethanes': ParameterInfo(0.74, 124, 0, 80, 'μg/L', 'Disinfection byproducts'),
        'Turbidity': ParameterInfo(1.45, 6.74, 0, 1, 'NTU', 'Water clarity measure')
    }
    
    # Color Configuration
    COLORS: Dict[str, str] = {
        'potable': '#28a745',
        'non_potable': '#dc3545',
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'info': '#17a2b8'
    }
    
    # Sample Data
    SAMPLE_DATA: Dict[str, Dict[str, float]] = {
    'potable': {
        'ph': 7.4,
        'Hardness': 90.0,
        'Solids': 320.0,
        'Chloramines': 3.0,
        'Sulfate': 200.0,
        'Conductivity': 380.0,
        'Organic_carbon': 3.5,
        'Trihalomethanes': 50.0,
        'Turbidity': 2.0
    },
    'non_potable': {
        'ph': 5.8,
        'Hardness': 250.0,
        'Solids': 25000.0,
        'Chloramines': 9.2,
        'Sulfate': 450.0,
        'Conductivity': 600.0,
        'Organic_carbon': 18.0,
        'Trihalomethanes': 95.0,
        'Turbidity': 5.8
    }
}
    
    # CSS Styles
    CUSTOM_CSS: str = """
    <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        .metric-card {
            background: linear-gradient(135deg, #f0f2f6 0%, #e8ecf0 100%);
            padding: 1.5rem;
            border-radius: 0.75rem;
            border-left: 4px solid #1f77b4;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
        }
        .prediction-result {
            padding: 1.5rem;
            border-radius: 0.75rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .potable {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-left: 4px solid #28a745;
        }
        .non-potable {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border-left: 4px solid #dc3545;
        }
        .sidebar-metric {
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 0.25rem;
            margin: 0.25rem 0;
        }
    </style>
    """
    
    @classmethod
    def get_parameter_display_name(cls, param: str) -> str:
        """Get display name for parameter"""
        return param.replace('_', ' ').title()
    
    @classmethod
    def get_model_config(cls, model_type: ModelType) -> Dict[str, Any]:
        """Get configuration for specific model type"""
        configs = {
            ModelType.RANDOM_FOREST: {
                'n_estimators': cls.MODEL.rf_estimators,
                'random_state': cls.MODEL.random_state
            },
            ModelType.LOGISTIC_REGRESSION: {
                'random_state': cls.MODEL.random_state,
                'max_iter': cls.MODEL.lr_max_iter
            },
            ModelType.SVM: {
                'random_state': cls.MODEL.random_state,
                'kernel': cls.MODEL.svm_kernel,
                'probability': True
            },
            ModelType.GRADIENT_BOOSTING: {
                'n_estimators': cls.MODEL.gb_estimators,
                'learning_rate': cls.MODEL.gb_learning_rate,
                'random_state': cls.MODEL.random_state
            }
        }
        return configs.get(model_type, {})
