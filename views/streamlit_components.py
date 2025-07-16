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
        <div style="
            background: linear-gradient(135deg, #f0f2f6 0%, #e8ecf0 100%);
            padding: 1.5rem;
            border-radius: 0.75rem;
            border-left: 4px solid {color_value};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        ">
            <h3 style="margin: 0; color: #333;">{self.title}</h3>
            <h2 style="margin: 0.5rem 0; color: {color_value};">{self.value}</h2>
            <p style="margin: 0; color: #666;">{self.description}</p>
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
                min_value=0.0, max_value=14.0, value=7.0, step=0.1,
                help="Acidity/alkalinity level (0-14)"
            )
            
            self.input_values['Hardness'] = st.number_input(
                "Hardness (mg/L)", 
                min_value=0.0, value=180.0, step=1.0,
                help="Mineral content in water"
            )
            
            self.input_values['Solids'] = st.number_input(
                "Total Dissolved Solids (ppm)", 
                min_value=0.0, value=15000.0, step=100.0,
                help="Total dissolved solids concentration"
            )
        
        with col2:
            self.input_values['Chloramines'] = st.number_input(
                "Chloramines (ppm)", 
                min_value=0.0, value=6.5, step=0.1,
                help="Disinfectant levels"
            )
            
            self.input_values['Sulfate'] = st.number_input(
                "Sulfate (mg/L)", 
                min_value=0.0, value=300.0, step=1.0,
                help="Sulfate concentration"
            )
            
            self.input_values['Conductivity'] = st.number_input(
                "Conductivity (μS/cm)", 
                min_value=0.0, value=400.0, step=1.0,
                help="Electrical conductivity"
            )
        
        with col3:
            self.input_values['Organic_carbon'] = st.number_input(
                "Organic Carbon (ppm)", 
                min_value=0.0, value=12.0, step=0.1,
                help="Organic carbon content"
            )
            
            self.input_values['Trihalomethanes'] = st.number_input(
                "Trihalomethanes (μg/L)", 
                min_value=0.0, value=65.0, step=0.1,
                help="Disinfection byproducts"
            )
            
            self.input_values['Turbidity'] = st.number_input(
                "Turbidity (NTU)", 
                min_value=0.0, value=3.5, step=0.1,
                help="Water clarity measure"
            )
        
        return self.input_values


class PredictionResultDisplay(BaseComponent):
    """Prediction result display component"""
    
    def __init__(self):
        super().__init__("Prediction Result")
    
    def render(self, result, **kwargs) -> None:
        """Render prediction result"""
        if result.is_potable:
            self._render_potable_result(result)
        else:
            self._render_non_potable_result(result)
        
        # Display additional information
        self._render_confidence_info(result)
    
    def _render_potable_result(self, result) -> None:
        """Render potable water result"""
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            padding: 1.5rem;
            border-radius: 0.75rem;
            border-left: 4px solid #28a745;
            margin: 1rem 0;
        ">
            <h3>✅ Water is POTABLE</h3>
            <p><strong>Confidence:</strong> {result.confidence:.1%}</p>
            <p><strong>Risk Level:</strong> {result.risk_level}</p>
            <p>This water sample is predicted to be safe for consumption.</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_non_potable_result(self, result) -> None:
        """Render non-potable water result"""
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            padding: 1.5rem;
            border-radius: 0.75rem;
            border-left: 4px solid #dc3545;
            margin: 1rem 0;
        ">
            <h3>⚠️ Water is NOT POTABLE</h3>
            <p><strong>Confidence:</strong> {result.confidence:.1%}</p>
            <p><strong>Risk Level:</strong> {result.risk_level}</p>
            <p>This water sample is predicted to be unsafe for consumption.</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_confidence_info(self, result) -> None:
        """Render confidence information"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Potable Probability", f"{result.probabilities[1]:.1%}")
        
        with col2:
            st.metric("Non-Potable Probability", f"{result.probabilities[0]:.1%}")


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


class ComponentFactory:
    """Factory for creating UI components"""
    
    @staticmethod
    def create_metric_card(title: str, value: str, description: str, color: str = "primary") -> MetricCard:
        """Create metric card"""
        return MetricCard(title, value, description, color)