"""
Prediction service with business logic and validation
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime

from core.base_classes import Observer, DataValidator
from models.ml_models import ModelManager
from utils.data_processor import DataProcessor
from config.settings import Settings


class PredictionResult:
    """Data class for prediction results"""
    
    def __init__(self, prediction: int, probabilities: np.ndarray, 
                 model_name: str, confidence: float, input_data: Dict[str, float],
                 timestamp: datetime = None):
        self.prediction = prediction
        self.probabilities = probabilities
        self.model_name = model_name
        self.confidence = confidence
        self.input_data = input_data
        self.timestamp = timestamp or datetime.now()
    
    @property
    def is_potable(self) -> bool:
        """Check if water is predicted as potable"""
        return self.prediction == 1
    
    @property
    def risk_level(self) -> str:
        """Get risk level based on confidence"""
        if self.confidence > 0.8:
            return "Low Risk" if self.is_potable else "High Risk"
        elif self.confidence > 0.6:
            return "Medium Risk"
        else:
            return "Uncertain"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'prediction': self.prediction,
            'is_potable': self.is_potable,
            'confidence': self.confidence,
            'risk_level': self.risk_level,
            'model_name': self.model_name,
            'probabilities': self.probabilities.tolist(),
            'input_data': self.input_data,
            'timestamp': self.timestamp.isoformat()
        }


class PredictionHistory:
    """Manages prediction history"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.history: List[PredictionResult] = []
    
    def add_prediction(self, result: PredictionResult) -> None:
        """Add prediction to history"""
        self.history.append(result)
        
        # Keep only recent predictions
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_recent_predictions(self, count: int = 10) -> List[PredictionResult]:
        """Get recent predictions"""
        return self.history[-count:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        if not self.history:
            return {}
        
        potable_count = sum(1 for result in self.history if result.is_potable)
        total_count = len(self.history)
        
        return {
            'total_predictions': total_count,
            'potable_predictions': potable_count,
            'non_potable_predictions': total_count - potable_count,
            'potable_percentage': (potable_count / total_count) * 100,
            'average_confidence': np.mean([result.confidence for result in self.history])
        }


class WaterQualityAnalyzer:
    """Analyzes water quality parameters"""
    
    @staticmethod
    def analyze_parameters(input_data: Dict[str, float]) -> Dict[str, Any]:
        """Analyze water quality parameters"""
        analysis = {
            'overall_quality': 'Good',
            'warnings': [],
            'recommendations': []
        }
        
        # pH recommendations
        if 'ph' in input_data:
            ph = input_data['ph']
            if ph < 6.5:
                analysis['warnings'].append("pH is too acidic")
                analysis['recommendations'].append("Consider pH adjustment - water is too acidic")
            elif ph > 8.5:
                analysis['warnings'].append("pH is too alkaline")
                analysis['recommendations'].append("Consider pH adjustment - water is too alkaline")
        
        # Chloramines recommendations
        if 'Chloramines' in input_data and input_data['Chloramines'] > 4:
            analysis['warnings'].append("High chloramine levels detected")
            analysis['recommendations'].append("High chloramine levels detected - consider additional filtration")
        
        # Turbidity recommendations
        if 'Turbidity' in input_data and input_data['Turbidity'] > 1:
            analysis['warnings'].append("High turbidity detected")
            analysis['recommendations'].append("High turbidity detected - water filtration recommended")
        
        # Determine overall quality
        if len(analysis['warnings']) > 3:
            analysis['overall_quality'] = 'Poor'
        elif len(analysis['warnings']) > 1:
            analysis['overall_quality'] = 'Fair'
        
        if not analysis['recommendations']:
            analysis['recommendations'].append("Water parameters are within acceptable ranges")
        
        return analysis


class PredictionService(Observer):
    """Main prediction service with comprehensive functionality"""
    
    def __init__(self, model_manager: ModelManager, data_processor: DataProcessor):
        self.model_manager = model_manager
        self.data_processor = data_processor
        self.history = PredictionHistory()
        self.logger = logging.getLogger(__name__)
        
        # Attach as observer to model manager
        self.model_manager.attach(self)
    
    def update(self, subject, event: str, data: Any = None) -> None:
        """Observer update method"""
        if event == "model_trained":
            self.logger.info(f"Model trained: {data.get('model_name')} with accuracy: {data.get('accuracy'):.4f}")
        elif event == "training_completed":
            self.logger.info(f"Training completed. Best model: {data.get('best_model')}")
    
    def predict_water_quality(self, input_data: Dict[str, float]) -> PredictionResult:
        """Comprehensive water quality prediction"""
        try:
            # Validate input data
            is_valid, errors = DataValidator.validate_water_parameters(input_data)
            if not is_valid:
                raise ValueError(f"Invalid input data: {', '.join(errors)}")
            
            # Make prediction
            prediction, probabilities, model_name = self.model_manager.predict(input_data)
            
            # Calculate confidence
            confidence = max(probabilities)
            
            # Create prediction result
            result = PredictionResult(
                prediction=prediction,
                probabilities=probabilities,
                model_name=model_name,
                confidence=confidence,
                input_data=input_data
            )
            
            # Add to history
            self.history.add_prediction(result)
            
            self.logger.info(f"Prediction made: {'Potable' if prediction == 1 else 'Non-Potable'} "
                           f"(Confidence: {confidence:.2%})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            raise
    
    def get_history_statistics(self) -> Dict[str, Any]:
        """Get prediction history statistics"""
        return self.history.get_statistics()
    
    def export_history(self) -> pd.DataFrame:
        """Export prediction history"""
        if not self.history.history:
            return pd.DataFrame()
        
        data = []
        for result in self.history.history:
            row = result.input_data.copy()
            row.update({
                'prediction': result.prediction,
                'is_potable': result.is_potable,
                'confidence': result.confidence,
                'risk_level': result.risk_level,
                'model_name': result.model_name,
                'timestamp': result.timestamp
            })
            data.append(row)
        
        return pd.DataFrame(data)