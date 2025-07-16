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
    PredictionResultDisplay, ParameterAnalysisTable, AlertManager
)


class WaterPotabilityApplication:
    """Main application class with comprehensive OOP design"""
    
    def __init__(self):
        self.setup_page()
        self.initialize_components()
        # Call setup_navigation after all page rendering methods are defined
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
        st.markdown(Settings.CUSTOM_CSS, unsafe_allow_html=True)
    
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
            self.analysis_table = ParameterAnalysisTable()
            
            logger.info("Application components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            st.error(f"Failed to initialize application: {e}")
    
    def render_header(self) -> None:
        """Render application header"""
        st.markdown(
            '<h1 class="main-header">💧 Water Potability Analysis System</h1>',
            unsafe_allow_html=True
        )
        st.markdown("### Comprehensive Analysis and Accurate Prediction of Water Quality Using Advanced Machine Learning"
        ) 

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
    
    def render_overview_page(self, **kwargs) -> None:
        """Render overview page"""
        st.header("Dataset Overview")
        
        # Load and process data
        df = self.data_processor.load_data()
        if df is None:
            AlertManager.show_error("Failed to load data. Please check your internet connection.")
            return
        
        # Calculate statistics
        stats = self.data_processor.calculate_statistics(df)
        
        if not stats:
            AlertManager.show_warning("Unable to calculate statistics")
            return
        
        # Display key metrics
        self._render_overview_metrics(stats)
        
        # Display charts
        self._render_overview_charts(df, stats)
        
        # Display data sample
        st.subheader("Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Display parameter information
        self._render_parameter_info()
    
    def _render_overview_metrics(self, stats: Dict[str, Any]) -> None:
        """Render overview metrics"""
        basic_info = stats.get('basic_info', {})
        class_balance = stats.get('statistical_analysis', {}).get('class_balance', {})
        data_quality = stats.get('data_quality', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            metric_card = ComponentFactory.create_metric_card(
                "Total Samples",
                f"{basic_info.get('total_samples', 0):,}",
                "Water quality records"
            )
            metric_card.render()
        
        with col2:
            metric_card = ComponentFactory.create_metric_card(
                "Parameters",
                str(len(Settings.PARAMETERS)),
                "Quality indicators"
            )
            metric_card.render()
        
        with col3:
            potable_pct = class_balance.get('potable_percentage', 0)
            metric_card = ComponentFactory.create_metric_card(
                "Potable Water",
                f"{potable_pct:.1f}%",
                "Safe for consumption",
                "success"
            )
            metric_card.render()
        
        with col4:
            completeness = data_quality.get('missing_values', {}).get('overall_completeness', 0)
            metric_card = ComponentFactory.create_metric_card(
                "Data Quality",
                f"{completeness:.1f}%",
                "Data completeness",
                "info"
            )
            metric_card.render()
    
    def _render_overview_charts(self, df: pd.DataFrame, stats: Dict[str, Any]) -> None:
        """Render overview charts"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Potability distribution pie chart
            class_balance = stats.get('statistical_analysis', {}).get('class_balance', {})
            potable_count = class_balance.get('potable_count', 0)
            non_potable_count = class_balance.get('non_potable_count', 0)
            
            pie_chart = ComponentFactory.create_chart('pie', "Water Potability Distribution")
            pie_chart.render(
                data=None,
                values=[potable_count, non_potable_count],
                names=['Potable', 'Non-Potable'],
                colors={'Potable': Settings.COLORS['potable'], 'Non-Potable': Settings.COLORS['non_potable']}
            )
        
        with col2:
            # Parameter correlation chart
            correlations = stats.get('statistical_analysis', {}).get('correlation_analysis', {})
            if correlations:
                corr_df = pd.DataFrame([
                    {'Parameter': Settings.get_parameter_display_name(param), 'Correlation': corr}
                    for param, corr in correlations.items()
                ]).sort_values('Correlation', key=abs, ascending=False)
                
                bar_chart = ComponentFactory.create_chart('bar', "Parameter Correlation with Potability")
                bar_chart.render(corr_df, x='Correlation', y='Parameter')
    
    def _render_parameter_info(self) -> None:
        """Render parameter information table"""
        st.subheader("Parameter Information")
        
        info_data = []
        for param, info in Settings.PARAMETER_INFO.items():
            info_data.append({
                'Parameter': Settings.get_parameter_display_name(param),
                'Range': info.range_str,
                'Optimal': info.optimal_str,
                'Unit': info.unit,
                'Description': info.description
            })
        
        info_df = pd.DataFrame(info_data)
        st.dataframe(info_df, use_container_width=True)
    
    def render_data_analysis_page(self, **kwargs) -> None:
        """Render data analysis page"""
        st.header("Statistical Analysis")
        
        # Load data
        df = self.data_processor.load_data()
        if df is None:
            AlertManager.show_error("Failed to load data")
            return
        
        # Calculate statistics
        stats = self.data_processor.calculate_statistics(df)
        
        # Display statistical analysis
        self._render_statistical_tables(stats)
        
        # Display data quality analysis
        self._render_data_quality_analysis(stats)
        
        # Display feature engineering suggestions
        self._render_feature_engineering_suggestions(df)
    
    def _render_statistical_tables(self, stats: Dict[str, Any]) -> None:
        """Render statistical analysis tables"""
        st.subheader("Parameter Statistics")
        
        descriptive_stats = stats.get('statistical_analysis', {}).get('descriptive_stats', {})
        correlations = stats.get('statistical_analysis', {}).get('correlation_analysis', {})
        
        if descriptive_stats:
            stats_data = []
            for param, param_stats in descriptive_stats.items():
                stats_data.append({
                    'Parameter': Settings.get_parameter_display_name(param),
                    'Mean': f"{param_stats['mean']:.2f}",
                    'Std Dev': f"{param_stats['std']:.2f}",
                    'Min': f"{param_stats['min']:.2f}",
                    'Max': f"{param_stats['max']:.2f}",
                    'Skewness': f"{param_stats['skewness']:.3f}",
                    'Correlation': f"{correlations.get(param, 0):.3f}"
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
    
    def _render_data_quality_analysis(self, stats: Dict[str, Any]) -> None:
        """Render data quality analysis"""
        st.subheader("Data Quality Assessment")
        
        data_quality = stats.get('data_quality', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            completeness = data_quality.get('missing_values', {}).get('overall_completeness', 0)
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        with col2:
            duplicates = data_quality.get('duplicates', {})
            duplicate_pct = duplicates.get('duplicate_percentage', 0)
            st.metric("Duplicate Records", f"{duplicate_pct:.1f}%")
        
        with col3:
            outliers = data_quality.get('outliers', {})
            if outliers:
                avg_outlier_pct = np.mean([info['percentage'] for info in outliers.values()])
                st.metric("Avg Outlier Rate", f"{avg_outlier_pct:.1f}%")
    
    def _render_feature_engineering_suggestions(self, df: pd.DataFrame) -> None:
        """Render feature engineering suggestions"""
        st.subheader("Feature Engineering Suggestions")
        
        suggestions = self.data_processor.get_feature_engineering_suggestions(df)
        
        for i, suggestion in enumerate(suggestions, 1):
            st.write(f"{i}. {suggestion}")
    
    def render_visualizations_page(self, **kwargs) -> None:
        """Render visualizations page"""
        st.header("Data Visualizations")
        
        # Load data
        df = self.data_processor.load_data()
        if df is None:
            AlertManager.show_error("Failed to load data")
            return
        
        # Visualization type selector
        viz_type = st.selectbox(
            "Select Visualization Type:",
            ["Distribution Analysis", "Correlation Analysis", "Advanced Visualizations"]
        )
        
        if viz_type == "Distribution Analysis":
            self._render_distribution_visualizations(df)
        elif viz_type == "Correlation Analysis":
            self._render_correlation_visualizations(df)
        elif viz_type == "Advanced Visualizations":
            self._render_advanced_visualizations(df)
    
    def _render_distribution_visualizations(self, df: pd.DataFrame) -> None:
        """Render distribution visualizations"""
        col1, col2 = st.columns(2)
        
        with col1:
            # pH distribution histogram
            histogram = ComponentFactory.create_chart('histogram', "pH Distribution by Potability")
            histogram.render(df, x='ph', color='Potability', nbins=30)
        
        with col2:
            # Parameter selection for box plot
            selected_param = st.selectbox(
                "Select Parameter for Box Plot",
                options=Settings.PARAMETERS,
                format_func=Settings.get_parameter_display_name
            )
            
            # Create box plot data
            box_data = []
            for potability in [0, 1]:
                subset = df[df['Potability'] == potability]
                for value in subset[selected_param]:
                    box_data.append({
                        'Parameter': selected_param,
                        'Value': value,
                        'Potability': 'Potable' if potability == 1 else 'Non-Potable'
                    })
            
            box_df = pd.DataFrame(box_data)
            bar_chart = ComponentFactory.create_chart('bar', f"{Settings.get_parameter_display_name(selected_param)} Distribution")
            bar_chart.render(box_df, x='Potability', y='Value', color='Potability')
    
    def _render_correlation_visualizations(self, df: pd.DataFrame) -> None:
        """Render correlation visualizations"""
        st.subheader("Parameter Relationships")
        
        col1, col2 = st.columns(2)
        with col1:
            x_param = st.selectbox("X-axis Parameter", Settings.PARAMETERS, key="scatter_x")
        with col2:
            y_param = st.selectbox("Y-axis Parameter", Settings.PARAMETERS, index=1, key="scatter_y")
        
        # Create scatter plot data
        scatter_data = df.sample(min(1000, len(df)))  # Sample for performance
        
        import plotly.express as px
        fig = px.scatter(
            scatter_data,
            x=x_param, y=y_param,
            color='Potability',
            title=f"{Settings.get_parameter_display_name(x_param)} vs {Settings.get_parameter_display_name(y_param)}",
            color_discrete_map={0: Settings.COLORS['non_potable'], 1: Settings.COLORS['potable']}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_advanced_visualizations(self, df: pd.DataFrame) -> None:
        """Render advanced visualizations"""
        st.subheader("Advanced Analysis")
        
        # Correlation heatmap
        correlation_matrix = self.data_processor.get_correlation_matrix(df)
        
        import plotly.express as px
        fig = px.imshow(
            correlation_matrix,
            title="Parameter Correlation Heatmap",
            color_continuous_scale="Teal",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_ml_page(self, **kwargs) -> None:
        """Render machine learning page"""
        st.header("Machine Learning Models")
        
        # Load data
        df = self.data_processor.load_data()
        if df is None:
            AlertManager.show_error("Failed to load data")
            return
        
        # Prepare data
        X, y = self.data_processor.prepare_features_target(df)
        
        # Train models
        with st.spinner("Training models..."):
            try:
                models = self.model_manager.train_models(X, y)
                AlertManager.show_success("Models trained successfully!")
            except Exception as e:
                AlertManager.show_error(f"Error training models: {e}")
                return
        
        # Display model comparison
        st.subheader("Model Performance Comparison")
        comparison_df = self.model_manager.get_model_comparison()
        st.dataframe(comparison_df, use_container_width=True)
        
        # Display best model
        best_model_name, _ = self.model_manager.get_best_model()
        best_accuracy = self.model_manager.evaluation_results[best_model_name]['accuracy']
        AlertManager.show_success(f"🏆 Best performing model: **{best_model_name}** (Accuracy: {best_accuracy:.4f})")
        
        # Feature importance
        importance_df = self.model_manager.get_feature_importance()
        if importance_df is not None:
            st.subheader("Feature Importance")
            bar_chart = ComponentFactory.create_chart('bar', "Feature Importance for Water Potability Prediction")
            bar_chart.render(importance_df, x='Importance', y='Feature')
    
    def render_prediction_page(self, **kwargs) -> None:
        """Render prediction page"""
        st.header("Water Quality Prediction")
        st.write("Enter water quality parameters to predict if the water is safe for consumption.")
        
        # Load and train models if needed
        df = self.data_processor.load_data()
        if df is None:
            AlertManager.show_error("Failed to load data")
            return
        
        if not self.model_manager.models:
            with st.spinner("Loading prediction models..."):
                X, y = self.data_processor.prepare_features_target(df)
                self.model_manager.train_models(X, y)
        
        # Render parameter input form
        input_data = self.parameter_form.render()
        
        # Sample data buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("Load Potable Sample", type="secondary"):
                st.session_state.update(Settings.SAMPLE_DATA['potable'])
                st.rerun()
        
        with col2:
            if st.button("Load Non-Potable Sample", type="secondary"):
                st.session_state.update(Settings.SAMPLE_DATA['non_potable'])
                st.rerun()
        
        # Prediction
        if st.button("🔮 Predict Water Quality", type="primary"):
            try:
                # Make prediction
                result = self.prediction_service.predict_water_quality(input_data)
                
                # Display result
                self.result_display.render(result)
                
                # Display parameter analysis
                if result.analysis is not None:
                    self.analysis_table.render(result.analysis)
                
                # Show disclaimer
                AlertManager.show_disclaimer()
                
            except Exception as e:
                AlertManager.show_error(f"Error making prediction: {e}")
    
    def render_history_page(self, **kwargs) -> None:
        """Render prediction history page"""
        st.header("Prediction History")
        
        # Get history statistics
        history_stats = self.prediction_service.get_history_statistics()
        
        if not history_stats or history_stats['total_predictions'] == 0:
            st.info("No predictions made yet. Visit the Prediction page to make your first prediction.")
            return
        
        # Display history statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", history_stats['total_predictions'])
        
        with col2:
            st.metric("Potable Predictions", history_stats['potable_predictions'])
        
        with col3:
            st.metric("Non-Potable Predictions", history_stats['non_potable_predictions'])
        
        with col4:
            st.metric("Average Confidence", f"{history_stats['average_confidence']:.1%}")
        
        # Display recent predictions
        st.subheader("Recent Predictions")
        recent_predictions = self.prediction_service.history.get_recent_predictions(10)
        
        if recent_predictions:
            history_data = []
            for result in recent_predictions:
                history_data.append({
                    'Timestamp': result.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'Prediction': 'Potable' if result.is_potable else 'Non-Potable',
                    'Confidence': f"{result.confidence:.1%}",
                    'Risk Level': result.risk_level,
                    'Model': result.model_name
                })
            
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
        
        # Export option
        if st.button("Export History"):
            export_df = self.prediction_service.export_history()
            if not export_df.empty:
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="prediction_history.csv",
                    mime="text/csv"
                )
    
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
