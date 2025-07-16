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