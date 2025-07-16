"""
Main Streamlit application with advanced OOP architecture
"""
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import logging
from typing import Dict, Any, Optional

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config.settings import Settings
from utils.data_processor import DataProcessor
from models.ml_models import ModelManager
from services.prediction_service import PredictionService
from views.streamlit_components import (
    NavigationManager, ComponentFactory, ParameterInputForm,
    PredictionResultDisplay, AlertManager
)


class WaterPotabilityApplication:
    """Main application class with comprehensive OOP design"""
    
    def __init__(self):
        self.setup_page()
        self.initialize_components()
        self.setup_navigation()
    
    def setup_page(self) -> None:
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title=Settings.APP.page_title,
            page_icon=Settings.APP.page_icon,
            layout=Settings.APP.layout,
            initial_sidebar_state=Settings.APP.initial_sidebar_state
        )
        
        # Apply custom CSS
        st.markdown("""
        <style>
            .main-header {
                font-size: 3rem;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
                font-weight: bold;
            }
            .stApp {
                background-color: #f8f9fa;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_components(self) -> None:
        """Initialize application components"""
        try:
            # Initialize core components
            self.data_processor = DataProcessor()
            self.model_manager = ModelManager()
            self.prediction_service = PredictionService(self.model_manager, self.data_processor)
            
            # Initialize UI components
            self.parameter_form = ParameterInputForm()
            self.result_display = PredictionResultDisplay()
            
            logger.info("Application components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            st.error(f"Failed to initialize application: {e}")
    
    def setup_navigation(self) -> None:
        """Setup navigation manager"""
        self.nav_manager = NavigationManager()
        
        # Register pages
        self.nav_manager.register_page("Overview", self.render_overview_page, "📊")
        self.nav_manager.register_page("Data Analysis", self.render_data_analysis_page, "📈")
        self.nav_manager.register_page("Visualizations", self.render_visualizations_page, "🎯")
        self.nav_manager.register_page("Machine Learning", self.render_ml_page, "🤖")
        self.nav_manager.register_page("Prediction", self.render_prediction_page, "🔮")
        self.nav_manager.register_page("History", self.render_history_page, "📋")
    
    def render_header(self) -> None:
        """Render application header"""
        st.markdown(
            '<h1 class="main-header">💧 Water Potability Analysis System</h1>',
            unsafe_allow_html=True
        )
        st.markdown("### Comprehensive analysis and prediction of water quality using advanced machine learning")
    
    def render_sidebar_metrics(self, stats: Optional[Dict[str, Any]]) -> None:
        """Render sidebar metrics"""
        st.sidebar.markdown("### Key Statistics")
        
        if stats and 'basic_info' in stats:
            basic_info = stats['basic_info']
            st.sidebar.metric("Total Samples", f"{basic_info['total_samples']:,}")
            
            if 'statistical_analysis' in stats and 'class_balance' in stats['statistical_analysis']:
                class_balance = stats['statistical_analysis']['class_balance']
                potable_pct = class_balance.get('potable_percentage', 0)
                st.sidebar.metric("Potable Water", f"{potable_pct:.1f}%")
            
            st.sidebar.metric("Parameters", len(Settings.PARAMETERS))
        
        # Prediction history stats
        history_stats = self.prediction_service.get_history_statistics()
        if history_stats:
            st.sidebar.markdown("### Prediction History")
            st.sidebar.metric("Total Predictions", history_stats['total_predictions'])
            if history_stats['total_predictions'] > 0:
                st.sidebar.metric("Avg Confidence", f"{history_stats['average_confidence']:.1%}")
    
    def run(self) -> None:
        """Main application runner"""
        try:
            # Render header
            self.render_header()
            
            # Load data for sidebar metrics
            df = self.data_processor.load_data()
            stats = None
            if df is not None:
                stats = self.data_processor.calculate_statistics(df)
            
            # Render sidebar and get selected page
            selected_page = self.nav_manager.render_sidebar_navigation()
            self.render_sidebar_metrics(stats)
            
            # Render selected page
            if selected_page:
                self.nav_manager.render_current_page()
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            AlertManager.show_error(f"Application error: {e}")


def main():
    """Main function to run the application"""
    try:
        app = WaterPotabilityApplication()
        app.run()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        st.error(f"Failed to start application: {e}")


if __name__ == "__main__":
    main()