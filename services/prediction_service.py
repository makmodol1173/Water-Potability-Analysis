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
                 analysis: pd.DataFrame, timestamp: datetime = None):
        self.prediction = prediction
        self.probabilities = probabilities
        self.model_name = model_name
        self.confidence = confidence
        self.input_data = input_data
        self.analysis = analysis
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
    
    def export_history(self) -> pd.DataFrame:
        """Export history as DataFrame"""
        if not self.history:
            return pd.DataFrame()
        
        data = []
        for result in self.history:
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


class WaterQualityAnalyzer:
    """Analyzes water quality parameters"""
    
    @staticmethod
    def analyze_parameters(input_data: Dict[str, float]) -> Dict[str, Any]:
        """Analyze water quality parameters"""
        analysis = {
            'overall_quality': 'Good',
            'critical_parameters': [],
            'warnings': [],
            'recommendations': []
        }
        
        critical_count = 0
        warning_count = 0
        
        for param, value in input_data.items():
            if param in Settings.PARAMETER_INFO:
                info = Settings.PARAMETER_INFO[param]
                
                if value < info.optimal_min or value > info.optimal_max:
                    if value < info.range_min or value > info.range_max:
                        analysis['critical_parameters'].append(param)
                        critical_count += 1
                    else:
                        analysis['warnings'].append(f"{Settings.get_parameter_display_name(param)} is outside optimal range")
                        warning_count += 1
        
        # Determine overall quality
        if critical_count > 0:
            analysis['overall_quality'] = 'Poor'
        elif warning_count > 3:
            analysis['overall_quality'] = 'Fair'
        elif warning_count > 0:
            analysis['overall_quality'] = 'Good'
        else:
            analysis['overall_quality'] = 'Excellent'
        
        # Generate recommendations
        analysis['recommendations'] = WaterQualityAnalyzer._generate_recommendations(input_data)
        
        return analysis
    
    @staticmethod
    def _generate_recommendations(input_data: Dict[str, float]) -> List[str]:
        """Generate recommendations based on parameter values"""
        recommendations = []
        
        # pH recommendations
        if 'ph' in input_data:
            ph = input_data['ph']
            if ph < 6.5:
                recommendations.append("Consider pH adjustment - water is too acidic")
            elif ph > 8.5:
                recommendations.append("Consider pH adjustment - water is too alkaline")
        
        # Chloramines recommendations
        if 'Chloramines' in input_data:
            chloramines = input_data['Chloramines']
            if chloramines > 4:
                recommendations.append("High chloramine levels detected - consider additional filtration")
        
        # Turbidity recommendations
        if 'Turbidity' in input_data:
            turbidity = input_data['Turbidity']
            if turbidity > 1:
                recommendations.append("High turbidity detected - water filtration recommended")
        
        # Organic carbon recommendations
        if 'Organic_carbon' in input_data:
            organic_carbon = input_data['Organic_carbon']
            if organic_carbon > 2:
                recommendations.append("High organic carbon levels - consider activated carbon filtration")
        
        if not recommendations:
            recommendations.append("Water parameters are within acceptable ranges")
        
        return recommendations


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
    
    def predict_water_quality(self, input_data: Dict[str, float], 
                            include_analysis: bool = True) -> PredictionResult:
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
            
            # Perform parameter analysis if requested
            analysis = None
            if include_analysis:
                analysis = self.data_processor.get_parameter_analysis(input_data)
            
            # Create prediction result
            result = PredictionResult(
                prediction=prediction,
                probabilities=probabilities,
                model_name=model_name,
                confidence=confidence,
                input_data=input_data,
                analysis=analysis
            )
            
            # Add to history
            self.history.add_prediction(result)
            
            self.logger.info(f"Prediction made: {'Potable' if prediction == 1 else 'Non-Potable'} "
                           f"(Confidence: {confidence:.2%})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            raise
    
    def batch_predict(self, input_data_list: List[Dict[str, float]]) -> List[PredictionResult]:
        """Batch prediction for multiple samples"""
        results = []
        
        for input_data in input_data_list:
            try:
                result = self.predict_water_quality(input_data, include_analysis=False)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error in batch prediction for {input_data}: {e}")
                continue
        
        return results
    
    def get_prediction_confidence_analysis(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """Get detailed confidence analysis for prediction"""
        try:
            # Get predictions from all models
            all_predictions = {}
            all_confidences = {}
            
            for model_name, model in self.model_manager.models.items():
                input_df = pd.DataFrame([input_data])
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0]
                
                all_predictions[model_name] = pred
                all_confidences[model_name] = max(prob)
            
            # Calculate consensus
            potable_votes = sum(1 for pred in all_predictions.values() if pred == 1)
            total_votes = len(all_predictions)
            consensus_strength = max(potable_votes, total_votes - potable_votes) / total_votes
            
            return {
                'model_predictions': all_predictions,
                'model_confidences': all_confidences,
                'consensus_strength': consensus_strength,
                'agreement_level': 'High' if consensus_strength > 0.8 else 'Medium' if consensus_strength > 0.6 else 'Low'
            }
            
        except Exception as e:
            self.logger.error(f"Error in confidence analysis: {e}")
            return {}
    
    def get_similar_cases(self, input_data: Dict[str, float], n_similar: int = 5) -> List[Dict[str, Any]]:
        """Find similar cases from prediction history"""
        if not self.history.history:
            return []
        
        # Calculate similarity based on Euclidean distance
        similarities = []
        
        for historical_result in self.history.history:
            distance = 0
            for param in Settings.PARAMETERS:
                if param in input_data and param in historical_result.input_data:
                    # Normalize by parameter range
                    param_info = Settings.PARAMETER_INFO.get(param)
                    if param_info:
                        param_range = param_info.range_max - param_info.range_min
                        normalized_diff = abs(input_data[param] - historical_result.input_data[param]) / param_range
                        distance += normalized_diff ** 2
            
            distance = np.sqrt(distance)
            similarities.append({
                'result': historical_result,
                'similarity': 1 / (1 + distance)  # Convert distance to similarity
            })
        
        # Sort by similarity and return top n
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return [{
            'input_data': sim['result'].input_data,
            'prediction': sim['result'].prediction,
            'confidence': sim['result'].confidence,
            'similarity': sim['similarity']
        } for sim in similarities[:n_similar]]
    
    def generate_report(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive water quality report"""
        try:
            # Make prediction
            prediction_result = self.predict_water_quality(input_data)
            
            # Analyze water quality
            quality_analysis = WaterQualityAnalyzer.analyze_parameters(input_data)
            
            # Get confidence analysis
            confidence_analysis = self.get_prediction_confidence_analysis(input_data)
            
            # Get similar cases
            similar_cases = self.get_similar_cases(input_data)
            
            # Compile report
            report = {
                'prediction_result': prediction_result.to_dict(),
                'quality_analysis': quality_analysis,
                'confidence_analysis': confidence_analysis,
                'similar_cases': similar_cases,
                'generated_at': datetime.now().isoformat(),
                'report_id': f"WQ_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            raise
    
    def get_history_statistics(self) -> Dict[str, Any]:
        """Get prediction history statistics"""
        return self.history.get_statistics()
    
    def export_history(self) -> pd.DataFrame:
        """Export prediction history"""
        return self.history.export_history()
