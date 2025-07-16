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
    
    def notify(self, event: str, data: Any = None) -> None:
        """Notify all observers"""
        for observer in self._observers:
            observer.update(self, event, data)


class DataValidator:
    """Data validation utility class"""
    
    @staticmethod
    def validate_water_parameters(data: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate water quality parameters"""
        errors = []
        
        if 'ph' in data and not (0 <= data['ph'] <= 14):
            errors.append("pH must be between 0 and 14")
        
        for param in ['Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 
                      'Organic_carbon', 'Trihalomethanes', 'Turbidity']:
            if param in data and data[param] < 0:
                errors.append(f"{param} cannot be negative")
        
        return len(errors) == 0, errors