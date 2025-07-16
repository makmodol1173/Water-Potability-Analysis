"""
Machine learning models with advanced OOP design patterns
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from typing import Dict, Any, List, Tuple, Optional
import logging

from core.base_classes import BaseModel, IModelManager, Subject
from config.settings import Settings, ModelType


class ModelFactory:
    """Factory pattern for creating ML models"""
    
    @staticmethod
    def create_model(model_type: ModelType, **kwargs) -> BaseModel:
        """Create a model based on type"""
        # Will be implemented with individual models
        pass


class ModelEvaluator:
    """Model evaluation utility class"""
    
    @staticmethod
    def evaluate_model(model: BaseModel, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'roc_auc': roc_auc_score(y_test, probabilities[:, 1]),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
            'classification_report': classification_report(
                y_test, predictions, 
                target_names=['Non-Potable', 'Potable'],
                output_dict=True
            )
        }
        
        return metrics
    
    @staticmethod
    def cross_validate_model(model: BaseModel, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation"""
        cv_scores = cross_val_score(model.create_model(), X, y, cv=cv, scoring='accuracy')
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }


class ModelManager(IModelManager, Subject):
    """Enhanced model manager with factory pattern and comprehensive evaluation"""
    
    def __init__(self):
        super().__init__()
        self.models: Dict[str, BaseModel] = {}
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None
        self.logger = logging.getLogger(__name__)
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train all available models - to be implemented"""
        pass
    
    def predict(self, input_data: Dict[str, float]) -> Tuple[int, np.ndarray, str]:
        """Make prediction using best model - to be implemented"""
        pass
    
    def get_best_model(self) -> Tuple[str, BaseModel]:
        """Get the best performing model"""
        if not self.best_model_name:
            raise ValueError("No models have been trained")
        
        return self.best_model_name, self.models[self.best_model_name]
    
    # Add to existing ml_models.py

class RandomForestModel(BaseModel):
    """Random Forest model implementation"""
    
    def create_model(self) -> RandomForestClassifier:
        """Create Random Forest model"""
        return RandomForestClassifier(
            n_estimators=Settings.MODEL.rf_estimators,
            random_state=Settings.MODEL.random_state
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the Random Forest model"""
        self.model = self.create_model()
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        if not self.is_trained:
            return {}
        
        importance_dict = {}
        for i, feature in enumerate(Settings.PARAMETERS):
            importance_dict[feature] = self.model.feature_importances_[i]
        
        return importance_dict


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model implementation"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.scaler = StandardScaler()
    
    def create_model(self) -> LogisticRegression:
        """Create Logistic Regression model"""
        return LogisticRegression(
            random_state=Settings.MODEL.random_state,
            max_iter=Settings.MODEL.lr_max_iter
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the Logistic Regression model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model = self.create_model()
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class SVMModel(BaseModel):
    """Support Vector Machine model implementation"""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.scaler = StandardScaler()
    
    def create_model(self) -> SVC:
        """Create SVM model"""
        return SVC(
            random_state=Settings.MODEL.random_state,
            kernel='rbf',
            probability=True
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the SVM model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model = self.create_model()
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting model implementation"""
    
    def create_model(self) -> GradientBoostingClassifier:
        """Create Gradient Boosting model"""
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=Settings.MODEL.random_state
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the Gradient Boosting model"""
        self.model = self.create_model()
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)


# Update ModelFactory
class ModelFactory:
    """Factory pattern for creating ML models"""
    
    @staticmethod
    def create_model(model_type: ModelType, **kwargs) -> BaseModel:
        """Create a model based on type"""
        model_classes = {
            ModelType.RANDOM_FOREST: RandomForestModel,
            ModelType.LOGISTIC_REGRESSION: LogisticRegressionModel,
            ModelType.SVM: SVMModel,
            ModelType.GRADIENT_BOOSTING: GradientBoostingModel
        }
        
        model_class = model_classes.get(model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model_class(model_type.value, **kwargs)
    
    # Complete the ModelManager class

def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Train all available models"""
    try:
        self.logger.info("Starting model training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Settings.DATA.test_size,
            random_state=Settings.DATA.random_state,
            stratify=y
        )
        
        # Train models
        model_types = [ModelType.RANDOM_FOREST, ModelType.LOGISTIC_REGRESSION, 
                      ModelType.SVM, ModelType.GRADIENT_BOOSTING]
        
        results = {}
        
        for model_type in model_types:
            self.logger.info(f"Training {model_type.value}")
            
            # Create and train model
            model = ModelFactory.create_model(model_type)
            model.train(X_train, y_train)
            
            # Evaluate model
            evaluation = ModelEvaluator.evaluate_model(model, X_test, y_test)
            cv_results = ModelEvaluator.cross_validate_model(model, X_train, y_train)
            
            # Store results
            self.models[model_type.value] = model
            self.evaluation_results[model_type.value] = {**evaluation, **cv_results}
            
            results[model_type.value] = {
                'model': model,
                'evaluation': evaluation,
                'cross_validation': cv_results
            }
            
            self.notify("model_trained", {
                "model_name": model_type.value,
                "accuracy": evaluation['accuracy']
            })
        
        # Determine best model
        self.best_model_name = max(
            self.evaluation_results.keys(),
            key=lambda k: self.evaluation_results[k]['accuracy']
        )
        
        self.notify("training_completed", {
            "best_model": self.best_model_name,
            "models_trained": len(results)
        })
        
        return results
        
    except Exception as e:
        self.logger.error(f"Error in model training: {e}")
        raise

def predict(self, input_data: Dict[str, float]) -> Tuple[int, np.ndarray, str]:
    """Make prediction using best model"""
    if not self.best_model_name or self.best_model_name not in self.models:
        raise ValueError("No trained models available")
    
    best_model = self.models[self.best_model_name]
    
    # Prepare input data
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = best_model.predict(input_df)[0]
    probabilities = best_model.predict_proba(input_df)[0]
    
    return prediction, probabilities, self.best_model_name

def get_model_comparison(self) -> pd.DataFrame:
    """Get model performance comparison"""
    comparison_data = []
    
    for model_name, results in self.evaluation_results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{results['accuracy']:.4f}",
            'ROC AUC': f"{results['roc_auc']:.4f}",
            'CV Score': f"{results['cv_mean']:.4f} ± {results['cv_std']:.4f}"
        })
    
    return pd.DataFrame(comparison_data)

def get_feature_importance(self, model_name: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Get feature importance for specified model"""
    if model_name is None:
        model_name = self.best_model_name
    
    if model_name not in self.models:
        return None
    
    model = self.models[model_name]
    
    if hasattr(model, 'get_feature_importance'):
        importance_dict = model.get_feature_importance()
        importance_df = pd.DataFrame([
            {'Feature': Settings.get_parameter_display_name(feature), 'Importance': importance}
            for feature, importance in importance_dict.items()
        ]).sort_values('Importance', ascending=True)
        
        return importance_df
    
    return None