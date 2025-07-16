"""
Base classes and interfaces for the Water Potability Analysis System
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class IDataProcessor(ABC):
    """Interface for data processing operations"""
    
    @abstractmethod
    def load_data(self) -> Optional[pd.DataFrame]:
        """Load data from source"""
        pass
    
    @abstractmethod
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data"""
        pass
    
    @abstractmethod
    def calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for the data"""
        pass


class IModelManager(ABC):
    """Interface for model management operations"""
    
    @abstractmethod
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train machine learning models"""
        pass
    
    @abstractmethod
    def predict(self, input_data: Dict[str, float]) -> Tuple[int, np.ndarray, str]:
        """Make predictions using trained models"""
        pass
    
    @abstractmethod
    def get_best_model(self) -> Tuple[str, Any]:
        """Get the best performing model"""
        pass


class IVisualizer(ABC):
    """Interface for visualization operations"""
    
    @abstractmethod
    def create_chart(self, chart_type: str, data: Any, **kwargs) -> Any:
        """Create a chart of specified type"""
        pass


class BaseAnalyzer(ABC):
    """Base class for data analyzers"""
    
    def __init__(self, name: str):
        self.name = name
        self._results = {}
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform analysis on data"""
        pass
    
    def get_results(self) -> Dict[str, Any]:
        """Get analysis results"""
        return self._results
    
    def clear_results(self) -> None:
        """Clear stored results"""
        self._results = {}


class BaseModel(ABC):
    """Base class for machine learning models"""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.is_trained = False
        self.performance_metrics = {}
        self.config = kwargs
    
    @abstractmethod
    def create_model(self) -> Any:
        """Create the underlying ML model"""
        pass
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get model performance metrics"""
        return self.performance_metrics
    
    def set_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Set model performance metrics"""
        self.performance_metrics = metrics


class Observer(ABC):
    """Observer interface for the Observer pattern"""
    
    @abstractmethod
    def update(self, subject: 'Subject', event: str, data: Any = None) -> None:
        """Update method called by subject"""
        pass


class Subject:
    """Subject class for the Observer pattern"""
    
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer) -> None:
        """Attach an observer"""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        """Detach an observer"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self, event: str, data: Any = None) -> None:
        """Notify all observers"""
        for observer in self._observers:
            observer.update(self, event, data)


class Singleton:
    """Singleton metaclass"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DataValidator:
    """Data validation utility class"""
    
    @staticmethod
    def validate_water_parameters(data: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate water quality parameters"""
        errors = []
        
        # pH validation
        if 'ph' in data:
            if not (0 <= data['ph'] <= 14):
                errors.append("pH must be between 0 and 14")
        
        # Hardness validation
        if 'Hardness' in data:
            if data['Hardness'] < 0:
                errors.append("Hardness cannot be negative")
        
        # Solids validation
        if 'Solids' in data:
            if data['Solids'] < 0:
                errors.append("Total dissolved solids cannot be negative")
        
        # Chloramines validation
        if 'Chloramines' in data:
            if data['Chloramines'] < 0:
                errors.append("Chloramines cannot be negative")
        
        # Sulfate validation
        if 'Sulfate' in data:
            if data['Sulfate'] < 0:
                errors.append("Sulfate cannot be negative")
        
        # Conductivity validation
        if 'Conductivity' in data:
            if data['Conductivity'] < 0:
                errors.append("Conductivity cannot be negative")
        
        # Organic carbon validation
        if 'Organic_carbon' in data:
            if data['Organic_carbon'] < 0:
                errors.append("Organic carbon cannot be negative")
        
        # Trihalomethanes validation
        if 'Trihalomethanes' in data:
            if data['Trihalomethanes'] < 0:
                errors.append("Trihalomethanes cannot be negative")
        
        # Turbidity validation
        if 'Turbidity' in data:
            if data['Turbidity'] < 0:
                errors.append("Turbidity cannot be negative")
        
        return len(errors) == 0, errors
