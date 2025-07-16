"""
Streamlit UI components with advanced OOP design
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
import logging

from config.settings import Settings
from services.prediction_service import PredictionResult, PredictionService


class BaseComponent(ABC):
    """Base class for Streamlit components"""
    
    def __init__(self, title: str):
        self.title = title
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def render(self, **kwargs) -> Any:
        """Render the component"""
        pass
    
    def render_header(self) -> None:
        """Render component header"""
        st.subheader(self.title)


class MetricCard(BaseComponent):
    """Metric card component"""
    
    def __init__(self, title: str, value: str, description: str, color: str = "primary"):
        super().__init__(title)
        self.value = value
        self.description = description
        self.color = color
    
    def render(self, **kwargs) -> None:
        """Render metric card"""
        color_value = Settings.COLORS.get(self.color, Settings.COLORS['primary'])
        
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {color_value};">
            <h3>{self.title}</h3>
            <h2 style="color: {color_value};">{self.value}</h2>
            <p>{self.description}</p>
        </div>
        """, unsafe_allow_html=True)


class ParameterInputForm(BaseComponent):
    """Parameter input form component"""
    
    def __init__(self):
        super().__init__("Water Quality Parameters")
        self.input_values = {}
    
    def render(self, **kwargs) -> Dict[str, float]:
        """Render parameter input form"""
        self.render_header()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self.input_values['ph'] = st.number_input(
                "pH Level", 
                min_value=0.0, max_value=14.0, value=7.5, step=0.1,
                help="Acidity/alkalinity level (0-14)"
            )
            
            self.input_values['Hardness'] = st.number_input(
                "Hardness (mg/L)", 
                min_value=0.0, value=90.0, step=1.0,
                help="Mineral content in water"
            )
            
            self.input_values['Solids'] = st.number_input(
                "Total Dissolved Solids (ppm)", 
                min_value=0.0, value=320.0, step=100.0,
                help="Total dissolved solids concentration"
            )
        
        with col2:
            self.input_values['Chloramines'] = st.number_input(
                "Chloramines (ppm)", 
                min_value=0.0, value=3.5, step=0.1,
                help="Disinfectant levels"
            )
            
            self.input_values['Sulfate'] = st.number_input(
                "Sulfate (mg/L)", 
                min_value=0.0, value=200.0, step=1.0,
                help="Sulfate concentration"
            )
            
            self.input_values['Conductivity'] = st.number_input(
                "Conductivity (μS/cm)", 
                min_value=0.0, value=380.0, step=1.0,
                help="Electrical conductivity"
            )
        
        with col3:
            self.input_values['Organic_carbon'] = st.number_input(
                "Organic Carbon (ppm)", 
                min_value=0.0, value=3.5, step=0.1,
                help="Organic carbon content"
            )
            
            self.input_values['Trihalomethanes'] = st.number_input(
                "Trihalomethanes (μg/L)", 
                min_value=0.0, value=50.0, step=0.1,
                help="Disinfection byproducts"
            )
            
            self.input_values['Turbidity'] = st.number_input(
                "Turbidity (NTU)", 
                min_value=0.0, value=0.5, step=0.1,
                help="Water clarity measure"
            )
        
        return self.input_values
    
    def load_sample_data(self, sample_type: str) -> None:
        """Load sample data"""
        if sample_type in Settings.SAMPLE_DATA:
            sample_data = Settings.SAMPLE_DATA[sample_type]
            for key, value in sample_data.items():
                if key in self.input_values:
                    st.session_state[f"input_{key}"] = value


class PredictionResultDisplay(BaseComponent):
    """Prediction result display component"""
    
    def __init__(self):
        super().__init__("Prediction Result")
    
    def render(self, result: PredictionResult, **kwargs) -> None:
        """Render prediction result"""
        if result.is_potable:
            self._render_potable_result(result)
        else:
            self._render_non_potable_result(result)
        
        # Display additional information
        self._render_confidence_info(result)
        self._render_model_info(result)
    
    def _render_potable_result(self, result: PredictionResult) -> None:
        """Render potable water result"""
        st.markdown(f"""
        <div class="prediction-result potable">
            <h3>✅ Water is POTABLE</h3>
            <p><strong>Confidence:</strong> {result.confidence:.1%}</p>
            <p><strong>Risk Level:</strong> {result.risk_level}</p>
            <p>This water sample is predicted to be safe for consumption.</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_non_potable_result(self, result: PredictionResult) -> None:
        """Render non-potable water result"""
        st.markdown(f"""
        <div class="prediction-result non-potable">
            <h3>⚠️ Water is NOT POTABLE</h3>
            <p><strong>Confidence:</strong> {result.confidence:.1%}</p>
            <p><strong>Risk Level:</strong> {result.risk_level}</p>
            <p>This water sample is predicted to be unsafe for consumption.</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_confidence_info(self, result: PredictionResult) -> None:
        """Render confidence information"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Potable Probability", f"{result.probabilities[1]:.1%}")
        
        with col2:
            st.metric("Non-Potable Probability", f"{result.probabilities[0]:.1%}")
    
    def _render_model_info(self, result: PredictionResult) -> None:
        """Render model information"""
        st.info(f"**Model Used:** {result.model_name}")


class ParameterAnalysisTable(BaseComponent):
    """Parameter analysis table component"""
    
    def __init__(self):
        super().__init__("Parameter Analysis")
    
    def render(self, analysis_df: pd.DataFrame, **kwargs) -> None:
        """Render parameter analysis table"""
        self.render_header()
        
        # Style the dataframe
        styled_df = analysis_df.style.apply(self._color_status, subset=['Status'], axis=1)
        st.dataframe(styled_df, use_container_width=True)
    
    def _color_status(self, row) -> List[str]:
        """Color code status column"""
        status = row['Status']
        if '✅' in status:
            return ['background-color: #d4edda']
        elif '⚠️' in status:
            return ['background-color: #fff3cd']
        elif '❌' in status:
            return ['background-color: #f8d7da']
        else:
            return ['']


class ChartComponent(BaseComponent):
    """Base chart component"""
    
    def __init__(self, title: str, chart_type: str):
        super().__init__(title)
        self.chart_type = chart_type
    
    def render(self, data: Any, **kwargs) -> None:
        """Render chart"""
        self.render_header()
        
        try:
            chart = self._create_chart(data, **kwargs)
            st.plotly_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating chart: {e}")
            self.logger.error(f"Chart creation error: {e}")
    
    @abstractmethod
    def _create_chart(self, data: Any, **kwargs) -> go.Figure:
        """Create the chart"""
        pass


class PieChartComponent(ChartComponent):
    """Pie chart component"""
    
    def __init__(self, title: str = "Distribution"):
        super().__init__(title, "pie")
    
    def _create_chart(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create pie chart"""
        values = kwargs.get('values', [])
        names = kwargs.get('names', [])
        colors = kwargs.get('colors', None)
        
        fig = px.pie(
            values=values,
            names=names,
            title=self.title,
            color_discrete_map=colors
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig


class BarChartComponent(ChartComponent):
    """Bar chart component"""
    
    def __init__(self, title: str = "Comparison"):
        super().__init__(title, "bar")
    
    def _create_chart(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create bar chart"""
        x_col = kwargs.get('x', data.columns[0])
        y_col = kwargs.get('y', data.columns[1])
        color_col = kwargs.get('color', None)
        
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=self.title
        )
        return fig


class HistogramComponent(ChartComponent):
    """Histogram component"""
    
    def __init__(self, title: str = "Distribution"):
        super().__init__(title, "histogram")
    
    def _create_chart(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create histogram"""
        x_col = kwargs.get('x', data.columns[0])
        color_col = kwargs.get('color', None)
        nbins = kwargs.get('nbins', 30)
        
        fig = px.histogram(
            data,
            x=x_col,
            color=color_col,
            title=self.title,
            nbins=nbins
        )
        return fig


class ComponentFactory:
    """Factory for creating UI components"""
    
    @staticmethod
    def create_metric_card(title: str, value: str, description: str, color: str = "primary") -> MetricCard:
        """Create metric card"""
        return MetricCard(title, value, description, color)
    
    @staticmethod
    def create_chart(chart_type: str, title: str = "") -> ChartComponent:
        """Create chart component"""
        chart_classes = {
            'pie': PieChartComponent,
            'bar': BarChartComponent,
            'histogram': HistogramComponent
        }
        
        chart_class = chart_classes.get(chart_type)
        if chart_class is None:
            raise ValueError(f"Unknown chart type: {chart_type}")
        
        return chart_class(title)


class NavigationManager:
    """Manages navigation between pages"""
    
    def __init__(self):
        self.pages = {}
        self.current_page = None
    
    def register_page(self, name: str, render_func: Callable, icon: str = "") -> None:
        """Register a page"""
        self.pages[name] = {
            'render_func': render_func,
            'icon': icon,
            'display_name': f"{icon} {name}" if icon else name
        }
    
    def render_sidebar_navigation(self) -> str:
        """Render sidebar navigation"""
        st.sidebar.title("Navigation")
        
        page_options = [page['display_name'] for page in self.pages.values()]
        selected_display = st.sidebar.selectbox("Choose a section:", page_options)
        
        # Find the actual page name
        for name, page in self.pages.items():
            if page['display_name'] == selected_display:
                self.current_page = name
                return name
        
        return list(self.pages.keys())[0] if self.pages else None
    
    def render_current_page(self, **kwargs) -> None:
        """Render the current page"""
        if self.current_page and self.current_page in self.pages:
            render_func = self.pages[self.current_page]['render_func']
            render_func(**kwargs)
        else:
            st.error("Page not found")


class AlertManager:
    """Manages alerts and notifications"""
    
    @staticmethod
    def show_success(message: str, icon: str = "✅") -> None:
        """Show success alert"""
        st.success(f"{icon} {message}")
    
    @staticmethod
    def show_warning(message: str, icon: str = "⚠️") -> None:
        """Show warning alert"""
        st.warning(f"{icon} {message}")
    
    @staticmethod
    def show_error(message: str, icon: str = "❌") -> None:
        """Show error alert"""
        st.error(f"{icon} {message}")
    
    @staticmethod
    def show_info(message: str, icon: str = "ℹ️") -> None:
        """Show info alert"""
        st.info(f"{icon} {message}")
    
    @staticmethod
    def show_disclaimer() -> None:
        """Show prediction disclaimer"""
        st.info("⚠️ This prediction is based on machine learning analysis and should not replace professional water testing. Always consult certified water quality experts for definitive safety assessments.")
