import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs("Reports", exist_ok=True)
import pandas as pd
import sqlite3
import re  # Add missing regex import

from server.config import client, completion_model, embedding_model

# Remove RAG utilities import - reverting to IFC-only approach
# from utils.rag_utils import ecoform_rag_call

from scripts.core.acoustic_pipeline import run_pipeline, run_from_free_text
from scripts.core.llm_calls import extract_variables, build_answer

from scripts.core.recommend_recompute import recommend_recompute
from scripts.core.sql_query_handler import handle_llm_query as handle_llm_sql
from scripts.core.llm_acoustic_query_handler import handle_llm_query as handle_llm_acoustic
from scripts.core.fix_occ_import_error import InfoDock, EnhancedInfoDock
from scripts.progressive_acoustic_analysis import ProgressiveAcousticAnalysis

# ===== IMPROVED UI STYLING WITH BETTER CONTRAST AND FONT SIZES =====

# Enhanced Color Scheme with Better Contrast
COLOR_MAIN_BG = "#1A1A1A"           # Darker background for better contrast
COLOR_TITLE_BG = "#2D2D2D"          # Slightly lighter than main
COLOR_TITLE_TEXT = "#FFFFFF"        # Pure white for maximum contrast
COLOR_TAB_BG = "#FF6B35"           # Vibrant orange
COLOR_TAB_TEXT = "#FFFFFF"         # White text on orange
COLOR_DROPDOWN_FRAME = "#FF8C42"   # Lighter orange for frames
COLOR_DROPDOWN_BG = "#3A3A3A"      # Darker dropdown background
COLOR_INPUT_BG = "#FF6B35"         # Same as tab for consistency
COLOR_INPUT_TEXT = "#FFFFFF"       # White text
COLOR_CHAT_BG = "#2A2A2A"          # Slightly lighter than main
COLOR_CHAT_TEXT = "#FFFFFF"        # White text for readability
COLOR_SUCCESS = "#4CAF50"          # Green for success messages
COLOR_WARNING = "#FF9800"          # Orange for warnings
COLOR_ERROR = "#F44336"            # Red for errors
COLOR_INFO = "#2196F3"             # Blue for info

from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QTabWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
# Enhanced Font Definitions with Better Sizing
FONT_TITLE = QFont("Segoe UI", 24, QFont.Weight.Bold)
FONT_LABEL = QFont("Segoe UI", 14, QFont.Weight.DemiBold)
FONT_BODY = QFont("Segoe UI", 12)
FONT_BUTTON = QFont("Segoe UI", 13, QFont.Weight.Bold)
FONT_CHAT = QFont("Segoe UI", 13)
FONT_SMALL = QFont("Segoe UI", 10)
FONT_TAB = QFont("Segoe UI", 12, QFont.Weight.Bold)
FONT_DROPDOWN = QFont("Segoe UI", 12)                          # Dropdown text

os.environ["PYTHONOCC_DISPLAY"] = "pyqt6"
from OCC.Display.backend import load_backend
load_backend("pyqt6")
from OCC.Display.qtDisplay import qtViewer3d
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB

# Enhanced rendering imports
from OCC.Core.Graphic3d import (
    Graphic3d_NameOfMaterial, Graphic3d_TypeOfLightSource, 
    Graphic3d_TypeOfShadingModel, Graphic3d_TypeOfVisualization
)
from OCC.Core.AIS import AIS_Shape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Trsf, gp_Vec
from collections import defaultdict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget,
    QLabel, QComboBox, QPushButton, QTextEdit, QLineEdit, QFileDialog, QMessageBox,
    QDockWidget, QTextBrowser, QGroupBox, QSlider, QCheckBox, QSizePolicy, QStackedWidget  # Added enhanced widgets and size policy
)
from PyQt6.QtGui import QAction  # QAction is in QtGui, not QtWidgets

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt, QTimer

import ifcopenshell
import ifcopenshell.geom

from py2neo import Graph
import networkx as nx
import matplotlib.pyplot as plt

from scripts.core.neo4j_interface import Neo4jConnector
from neo4j import GraphDatabase

# Automatically register the folder where Qt WebEngine DLLs are stored
dll_path = os.path.join(sys.prefix, "Library", "bin")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(dll_path)  # OK Only works in Python 3.8+
else:
    os.environ["PATH"] = dll_path + os.pathsep + os.environ["PATH"]

try:
    from OCC.Display.backend import load_backend
    print("OK OCC module loaded successfully!")
except ModuleNotFoundError:
    print("X OCC module not found. Trying fallback import...")
    try:
        import OCC
        print("OK OCC base module available (but submodules missing). Try checking submodule availability.")
    except Exception as e:
        print("🚨 OCC could not be loaded at all:", e)
        print("Make sure 'pythonocc-core' is installed in the correct environment.")
        print("If you're using Conda:")
        print("  conda activate ifcenv")
        print("  conda install -c conda-forge pythonocc-core")

from neo4j import GraphDatabase

# Enhanced rendering functions
def enhance_viewer_rendering(viewer):
    """Apply enhanced rendering settings to the viewer"""
    try:
        # Set visualization type to shaded
        viewer._display.Context.SetDisplayMode(1, True)
        
        # Enable anti-aliasing
        viewer._display.Context.SetAntialiasing(True)
        
        # Set shading model to smooth
        viewer._display.Context.SetShadingModel(Graphic3d_TypeOfShadingModel.Graphic3d_TOSM_FRAGMENT)
        
        # Configure lighting for better quality
        viewer._display.Context.SetDefaultLights()
        
        # Set background color to dark
        bg_color = Quantity_Color(0.1, 0.1, 0.1, Quantity_TOC_RGB)
        viewer._display.View.SetBackgroundColor(bg_color)
        
        print("OK Enhanced rendering settings applied")
        
    except Exception as e:
        print(f"WARNING Could not apply all enhanced settings: {e}")

def improve_shape_tessellation(shape, quality=0.1):
    """Improve the tessellation quality of a shape"""
    try:
        # Create incremental mesh with better quality
        mesh = BRepMesh_IncrementalMesh(shape, quality, False, quality, True)
        mesh.Perform()
        
        if mesh.IsDone():
            print(f"OK Shape quality improved with tessellation quality: {quality}")
            return shape
        else:
            print(f"WARNING Could not improve shape quality")
            return shape
            
    except Exception as e:
        print(f"WARNING Error improving shape quality: {e}")
        return shape

def apply_enhanced_materials(viewer, ais_shape, element_type):
    """Apply enhanced materials based on element type"""
    try:
        # Enhanced material mapping
        material_map = {
            'IfcWall': Graphic3d_NameOfMaterial.Graphic3d_NOM_PLASTIC,
            'IfcSlab': Graphic3d_NameOfMaterial.Graphic3d_NOM_STONE,
            'IfcBeam': Graphic3d_NameOfMaterial.Graphic3d_NOM_METALIZED,
            'IfcColumn': Graphic3d_NameOfMaterial.Graphic3d_NOM_METALIZED,
            'IfcDoor': Graphic3d_NameOfMaterial.Graphic3d_NOM_WOOD,
            'IfcWindow': Graphic3d_NameOfMaterial.Graphic3d_NOM_GLASS,
            'IfcRoof': Graphic3d_NameOfMaterial.Graphic3d_NOM_STONE,
            'IfcStair': Graphic3d_NameOfMaterial.Graphic3d_NOM_STONE,
            'IfcFurniture': Graphic3d_NameOfMaterial.Graphic3d_NOM_WOOD,
            'IfcSanitaryTerminal': Graphic3d_NameOfMaterial.Graphic3d_NOM_PLASTIC,
        }
        
        material = material_map.get(element_type, Graphic3d_NameOfMaterial.Graphic3d_NOM_PLASTIC)
        viewer._display.Context.SetMaterial(ais_shape, material, False)
        
        # Set transparency for windows
        if element_type == 'IfcWindow':
            viewer._display.Context.SetTransparency(ais_shape, 0.7, False)
        
    except Exception as e:
        print(f"WARNING Could not apply enhanced material: {e}")

def create_enhanced_ifc_settings():
    """Create enhanced IFC geometry settings"""
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_PYTHON_OPENCASCADE, True)
    settings.set(settings.USE_WORLD_COORDS, True)
    
    # Only use settings that are guaranteed to exist
    # Note: INCLUDE_CURVES and SEW_SHELLS may not exist in all versions
    
    return settings

# Apartment overlay functions
def parse_apartment_info(space_name):
    """Parse apartment information from space name"""
    if not space_name:
        return None
    
    try:
        parts = space_name.split('_')
        if len(parts) >= 3:
            level = int(parts[0].replace('LVL', ''))
            apt_type = parts[1]  # 1B, 2B, 3B
            apt_number = int(parts[2])
            return {
                "level": level,
                "type": apt_type,
                "number": apt_number
            }
    except:
        pass
    return None

def get_apartment_color(apartment_type):
    """Get color for apartment type"""
    color_map = {
        "1B": (0.2, 0.6, 1.0),    # Blue
        "2B": (0.2, 0.8, 0.2),    # Green  
        "3B": (1.0, 0.4, 0.2),    # Orange
        # Also support alternative formats
        "1bed": (0.2, 0.6, 1.0),  # Blue
        "2bed": (0.2, 0.8, 0.2),  # Green
        "3bed": (1.0, 0.4, 0.2),  # Orange
        "1 bedroom": (0.2, 0.6, 1.0),  # Blue
        "2 bedroom": (0.2, 0.8, 0.2),  # Green
        "3 bedroom": (1.0, 0.4, 0.2),  # Orange
    }
    return color_map.get(apartment_type, (0.7, 0.7, 0.7))  # Default gray

def find_apartment_spaces(ifc_file):
    """Find all apartment spaces and their properties"""
    apartment_spaces = {}
    
    # Get all IfcSpace objects
    spaces = ifc_file.by_type("IfcSpace")
    
    for space in spaces:
        space_name = getattr(space, "Name", "")
        long_name = getattr(space, "LongName", "")
        
        # Try to parse apartment info from name
        apt_info = parse_apartment_info(space_name)
        if not apt_info:
            apt_info = parse_apartment_info(long_name)
        
        if apt_info:
            apartment_spaces[space.GlobalId] = {
                "space": space,
                "info": apt_info,
                "color": get_apartment_color(apt_info["type"])
            }
    
    return apartment_spaces

def find_elements_in_space(ifc_file, space_global_id):
    """Find all elements that belong to a specific space"""
    elements = []
    
    try:
        # Get the space object
        space = ifc_file.by_guid(space_global_id)
        
        # Method 1: Check IfcRelContainedInSpatialStructure
        if hasattr(space, "ContainsElements"):
            for rel in space.ContainsElements:
                for element in rel.RelatedElements:
                    elements.append(element)
        
        # Method 2: Check IfcRelSpaceBoundary
        if hasattr(space, "BoundedBy"):
            for rel in space.BoundedBy:
                if hasattr(rel, "RelatedBuildingElement"):
                    elements.append(rel.RelatedBuildingElement)
        
        # Method 3: Check by spatial structure hierarchy
        all_products = ifc_file.by_type("IfcProduct")
        for product in all_products:
            if hasattr(product, "ContainedInStructure"):
                for rel in product.ContainedInStructure:
                    if rel.RelatingStructure.GlobalId == space_global_id:
                        elements.append(product)
        
    except Exception as e:
        print(f"WARNING Error finding elements for space {space_global_id}: {e}")
    
    return elements

def create_apartment_overlay(viewer, ifc_file_path):
    """Create apartment color overlay for all elements"""
    try:
        # Load IFC file
        ifc_file = ifcopenshell.open(ifc_file_path)
        
        # Find apartment spaces
        apartment_spaces = find_apartment_spaces(ifc_file)
        print(f"BUILDING Found {len(apartment_spaces)} apartment spaces")
        
        # Create mapping of elements to apartment colors
        element_to_apartment = {}
        apartment_elements = defaultdict(list)
        
        # Process each apartment space
        for space_global_id, space_data in apartment_spaces.items():
            apt_info = space_data["info"]
            color = space_data["color"]
            
            print(f"ART Processing apartment {apt_info['type']} on level {apt_info['level']}")
            
            # Find elements in this space
            elements = find_elements_in_space(ifc_file, space_global_id)
            
            for element in elements:
                element_to_apartment[element.GlobalId] = {
                    "apartment_type": apt_info["type"],
                    "apartment_level": apt_info["level"],
                    "apartment_number": apt_info["number"],
                    "color": color
                }
                apartment_elements[apt_info["type"]].append(element)
            
            print(f"   PACKAGE Found {len(elements)} elements in apartment {apt_info['type']}")
        
        # Apply colors to viewer shapes
        apply_apartment_colors_to_viewer(viewer, element_to_apartment)
        
        # Return statistics
        stats = {
            "total_apartments": len(apartment_spaces),
            "total_elements": len(element_to_apartment),
            "apartment_counts": {apt_type: len(elements) for apt_type, elements in apartment_elements.items()}
        }
        
        return stats
        
    except Exception as e:
        print(f"X Error creating apartment overlay: {e}")
        return None

def apply_apartment_colors_to_viewer(viewer, element_to_apartment):
    """Apply apartment colors to the viewer shapes"""
    try:
        colored_count = 0
        
        # Iterate through all shapes in the viewer
        for ais_shape, ifc_element in viewer.shape_to_ifc.items():
            if hasattr(ifc_element, "GlobalId"):
                global_id = ifc_element.GlobalId
                
                if global_id in element_to_apartment:
                    apartment_data = element_to_apartment[global_id]
                    color = apartment_data["color"]
                    
                    # Apply color to the shape
                    color_obj = Quantity_Color(color[0], color[1], color[2], Quantity_TOC_RGB)
                    viewer._display.Context.SetColor(ais_shape, color_obj, False)
                    
                    colored_count += 1
        
        # Update the display
        viewer._display.Context.UpdateCurrentViewer()
        viewer._display.Repaint()
        
        print(f"OK Applied apartment colors to {colored_count} elements")
        
    except Exception as e:
        print(f"X Error applying colors to viewer: {e}")

def reset_apartment_colors(viewer):
    """Reset colors to original element type colors"""
    try:
        for ais_shape, ifc_element in viewer.shape_to_ifc.items():
            element_type = ifc_element.is_a()
            color = viewer.color_map.get(element_type, viewer.default_color)
            color_obj = Quantity_Color(color[0], color[1], color[2], Quantity_TOC_RGB)
            viewer._display.Context.SetColor(ais_shape, color_obj, False)
        
        viewer._display.Context.UpdateCurrentViewer()
        viewer._display.Repaint()
        print("OK Reset to original element colors")
        
    except Exception as e:
        print(f"X Error resetting colors: {e}")

def create_apartment_legend():
    """Create a legend showing apartment type colors"""
    legend_html = """
    <div style="background: #2C2C2C; padding: 15px; border-radius: 10px; color: white;">
        <h3 style="margin-top: 0; color: #FF9500;">Apartment Type Legend</h3>
        <div style="display: flex; flex-direction: column; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 20px; height: 20px; background: rgb(51, 153, 255); border-radius: 3px;"></div>
                <span>1 Bedroom (1B)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 20px; height: 20px; background: rgb(51, 204, 51); border-radius: 3px;"></div>
                <span>2 Bedroom (2B)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 20px; height: 20px; background: rgb(255, 102, 51); border-radius: 3px;"></div>
                <span>3 Bedroom (3B)</span>
            </div>
        </div>
    </div>
    """
    return legend_html

class Neo4jHelper:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="yourpassword"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_neighbors(self, global_id):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (a:IFCElement {GlobalId: $gid})-[r]->(b)
                RETURN b.GlobalId, type(r)
                """, gid=global_id
            )
            return [dict(record) for record in result]

# ---- Color and Font scheme ----
COLOR_TITLE_BG = "#2C2C2C"
COLOR_TITLE_TEXT = "#FDF6F6"
COLOR_TAB_BG = "#FF8A05"
COLOR_TAB_TEXT = "#FDF6F6"
COLOR_DROPDOWN_FRAME = "#E98801"
COLOR_DROPDOWN_BG = "#31353C"
COLOR_INPUT_BG = "#FF8F04"
COLOR_INPUT_TEXT = "#FDF6F6"
COLOR_CHAT_BG = "#2C2C2C"
COLOR_CHAT_TEXT = "#FDF6F6"
COLOR_MAIN_BG = "#121516"
BG_MAIN = "#181A1B"
BG_PANEL = "#23262B"
ACCENT = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF9500, stop:1 #FF7F50)"
INPUT_BG = "#2E3238"
BORDER_ACCENT = "#FF9500"
TEXT_MAIN = "#Fffafa"
TEXT_ACCENT = "#FFA040"

FONT_TITLE = QFont("Segoe UI", 20, QFont.Weight.Bold)
FONT_LABEL = QFont("Segoe UI", 12, QFont.Weight.DemiBold)
FONT_BODY = QFont("Segoe UI", 11)
FONT_BUTTON = QFont("Segoe UI", 12, QFont.Weight.Bold)
FONT_CHAT = QFont("Segoe UI", 14)  # Larger chat font

# ---- Styles ----
panel_style = f"background: {COLOR_DROPDOWN_BG}; border-radius: 12px; padding: 10px;"
btn_style = f"""
    QPushButton {{
        background: {COLOR_DROPDOWN_FRAME};
        color: {TEXT_MAIN};
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background: #FFA040;
        color: black;
    }}
"""
small_btn_style = f"""
    QPushButton {{
        background: {COLOR_DROPDOWN_FRAME};
        color: {TEXT_MAIN};
        border-radius: 6px;
        padding: 6px 12px;
        font-weight: bold;
        font-size: 11px;
    }}
    QPushButton:hover {{
        background: #FFA040;
        color: black;
    }}
"""
input_style = f"""
    QComboBox, QLineEdit {{
        background: {INPUT_BG};
        color: {TEXT_MAIN};
        border: 2px solid {BORDER_ACCENT};
        border-radius: 8px;
        padding: 6px;
    }}
"""
chat_style = f"""
    QTextEdit {{
        background: #24282c;
        color: {COLOR_INPUT_TEXT};
        border-radius: 12px;
        font-size: 18px;
        border: 1px solid {BORDER_ACCENT};
        padding: 10px;
        font-weight: normal;
    }}
    QLineEdit {{
        background: #24282c;
        color: {COLOR_INPUT_TEXT};
        border-radius: 8px;
        font-size: 16px;
        border: 1px solid {BORDER_ACCENT};
        padding: 8px;
        font-weight: normal;
    }}
"""


class Neo4jViewer(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Neo4j Viewer", parent)
        
        # Create a central widget to hold the content
        central_widget = QWidget()
        self.setWidget(central_widget)
        
        self.web_view = QWebEngineView()
        self.web_view.load(QUrl("http://localhost:7474/browser/"))

        self.graph_btn = QPushButton("Show Preprogrammed Graph")
        self.graph_btn.clicked.connect(self.show_custom_graph)

        layout = QVBoxLayout()
        layout.addWidget(self.web_view)
        layout.addWidget(self.graph_btn)
        central_widget.setLayout(layout)

    def show_custom_graph(self):
        win = QWebEngineView()
        win.setWindowTitle("Custom Graph")
        win.resize(1000, 700)
        win.load(QUrl.fromLocalFile('/absolute/path/to/your/graph.html'))
        win.show()


class MyViewer(qtViewer3d):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.shape_to_ifc = {}
        self.color_map = {
            'IfcWall': (0.9, 0.9, 0.9),        # Light grey - Walls
            'IfcSlab': (0.7, 0.7, 0.7),        # Medium grey - Floors/Slabs
            'IfcBeam': (0.5, 0.5, 0.5),        # Dark grey - Beams
            'IfcColumn': (0.3, 0.3, 0.3),      # Very dark grey - Columns
            'IfcDoor': (0.8, 0.8, 0.8),        # Light grey - Doors
            'IfcWindow': (0.6, 0.6, 0.6),      # Medium grey - Windows
            'IfcRoof': (0.4, 0.4, 0.4),        # Dark grey - Roof
            'IfcStair': (0.2, 0.2, 0.2),       # Very dark grey - Stairs
            'IfcFurniture': (0.75, 0.75, 0.75), # Light grey - Furniture
            'IfcSanitaryTerminal': (0.45, 0.45, 0.45), # Medium dark grey - Sanitary
            'IfcSpace': (0.8, 0.6, 0.8),       # Light purple - Spaces (semi-transparent)
        }
        self.default_color = (0.7, 0.7, 0.7)

        # The dock should be created and owned by the main window, not the viewer itself
        self.info_dock = None  # To be set externally from the main window
        
        # Enhanced rendering properties
        self.enhanced_rendering_enabled = False
        self.apartment_overlay_enabled = False
        self.current_ifc_path = None

    def enable_enhanced_rendering(self):
        """Enable enhanced rendering features"""
        try:
            enhance_viewer_rendering(self)
            self.enhanced_rendering_enabled = True
            print("OK Enhanced rendering enabled")
        except Exception as e:
            print(f"X Failed to enable enhanced rendering: {e}")

    def load_ifc_with_enhanced_rendering(self, path):
        """Load IFC file with enhanced rendering"""
        try:
            self.current_ifc_path = path
            
            # Use enhanced IFC settings
            settings = create_enhanced_ifc_settings()
            
            # Load the IFC file
            ifc_file = ifcopenshell.open(path)
            
            # Clear existing shapes
            self._display.Context.RemoveAll(False)
            self.shape_to_ifc.clear()
            
            # Define loadable element types
            loadable_types = [
                "IfcWall", "IfcSlab", "IfcBeam", "IfcColumn", "IfcDoor", "IfcWindow",
                "IfcRoof", "IfcStair", "IfcFurniture", "IfcSanitaryTerminal",
                "IfcRailing", "IfcCovering", "IfcPlate", "IfcMember", "IfcFooting",
                "IfcPile", "IfcBuildingElementProxy", "IfcDistributionElement",
                "IfcFlowTerminal", "IfcFlowSegment", "IfcFlowFitting", "IfcSpace"
            ]
            
            # Process each product
            products = ifc_file.by_type("IfcProduct")
            loaded_count = 0
            error_count = 0
            
            for product in products:
                try:
                    # Skip non-loadable element types
                    if product.is_a() not in loadable_types:
                        continue
                    
                    # Special handling for IfcSpace - create bounding box only if no representation exists
                    if product.is_a() == "IfcSpace":
                        # First try to create shape normally
                        if hasattr(product, "Representation") and product.Representation:
                            shape_result = ifcopenshell.geom.create_shape(settings, product)
                            # If normal shape creation fails, create bounding box
                            if not shape_result or not hasattr(shape_result, 'geometry') or shape_result.geometry is None:
                                shape_result = self.create_space_bounding_box(product, settings)
                        else:
                            # No representation, create bounding box
                            shape_result = self.create_space_bounding_box(product, settings)
                    else:
                        # Skip elements without representation
                        if not hasattr(product, "Representation") or not product.Representation:
                            continue
                        # Create shape with enhanced settings
                        shape_result = ifcopenshell.geom.create_shape(settings, product)
                    
                    if shape_result and hasattr(shape_result, 'geometry') and shape_result.geometry is not None:
                        # Improve tessellation quality for non-space elements
                        if product.is_a() != "IfcSpace":
                            improved_shape = improve_shape_tessellation(shape_result.geometry)
                        else:
                            improved_shape = shape_result.geometry
                        
                        # Create AIS shape
                        ais_shape = AIS_Shape(improved_shape)
                        
                        # Apply enhanced materials for non-space elements
                        element_type = product.is_a()
                        if element_type != "IfcSpace":
                            apply_enhanced_materials(self, ais_shape, element_type)
                        
                        # Apply color
                        color = self.color_map.get(element_type, self.default_color)
                        color_obj = Quantity_Color(color[0], color[1], color[2], Quantity_TOC_RGB)
                        self._display.Context.SetColor(ais_shape, color_obj, False)
                        
                        # For spaces, make them semi-transparent
                        if element_type == "IfcSpace":
                            self._display.Context.SetTransparency(ais_shape, 0.7, False)
                        
                        # Display the shape
                        self._display.Context.Display(ais_shape, False)
                        
                        # Store mapping
                        self.shape_to_ifc[ais_shape] = product
                        loaded_count += 1
                        
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:  # Only show first 5 errors to avoid spam
                        print(f"WARNING Error processing {product.is_a()}: {str(e)[:100]}...")
                    continue
            
            # Update display
            self._display.Context.UpdateCurrentViewer()
            self._display.FitAll()
            
            print(f"OK Loaded IFC with enhanced rendering: {loaded_count} elements (skipped {error_count} errors)")
            
            if loaded_count == 0:
                print("WARNING No geometry loaded. This might be due to:")
                print("   - IFC file has no geometric representation")
                print("   - Elements are not in supported types")
                print("   - Display driver issues")
                return False
            
            return True
            
        except Exception as e:
            print(f"X Error loading IFC with enhanced rendering: {e}")
            return False

    def create_apartment_overlay(self):
        """Create apartment color overlay"""
        if not self.current_ifc_path:
            print("X No IFC file loaded")
            return None
            
        try:
            stats = create_apartment_overlay(self, self.current_ifc_path)
            if stats:
                self.apartment_overlay_enabled = True
                print(f"OK Apartment overlay created: {stats}")
                return stats
            else:
                print("X Failed to create apartment overlay")
                return None
        except Exception as e:
            print(f"X Error creating apartment overlay: {e}")
            return None

    def reset_to_element_colors(self):
        """Reset colors to original element type colors"""
        try:
            reset_apartment_colors(self)
            self.apartment_overlay_enabled = False
            print("OK Reset to element type colors")
        except Exception as e:
            print(f"X Error resetting colors: {e}")

    def toggle_apartment_overlay(self):
        """Toggle between apartment overlay and element colors"""
        if self.apartment_overlay_enabled:
            self.reset_to_element_colors()
        else:
            self.create_apartment_overlay()

    def get_apartment_legend_html(self):
        """Get HTML for apartment legend"""
        return create_apartment_legend()

    def get_ifc_summary(self):
        result = []
        for ifc_elem in self.shape_to_ifc.values():
            entry = {
                "type": ifc_elem.is_a(),
                "name": getattr(ifc_elem, "Name", ""),
                "space": getattr(ifc_elem, "LongName", ""),
                "RT60": None,
                "SPL": None,
            }
            # extract RT60 / SPL from property sets
            if hasattr(ifc_elem, "IsDefinedBy"):
                for rel in ifc_elem.IsDefinedBy:
                    if rel.is_a("IfcRelDefinesByProperties"):
                        props = rel.RelatingPropertyDefinition
                        if props.is_a("IfcPropertySet"):
                            for prop in props.HasProperties:
                                if prop.Name.lower().startswith("rt60"):
                                    entry["RT60"] = getattr(prop.NominalValue, "wrappedValue", None)
                                if prop.Name.lower().startswith("spl"):
                                    entry["SPL"] = getattr(prop.NominalValue, "wrappedValue", None)
            result.append(entry)
        return result

    def extract_model_data(self):
        result = []
        for ifc_elem in self.shape_to_ifc.values():
            name = getattr(ifc_elem, "Name", "") or ""
            entry = {
                "element_type": ifc_elem.is_a(),
                "name": name,
                "floor_level": None,
                "apartment_type": None,
                "RT60": None,
                "SPL": None,
                "area_m2": None
            }

            # Parse apartment info from name string (e.g., "LVL2_3B_36")
            if isinstance(name, str) and "LVL" in name:
                try:
                    lvl_part = name.split("_")[0]
                    entry["floor_level"] = int(lvl_part.replace("LVL", ""))
                except:
                    pass
            if isinstance(name, str) and "_1B" in name:
                entry["apartment_type"] = "1Bed"
            elif "_2B" in name:
                entry["apartment_type"] = "2Bed"
            elif "_3B" in name:
                entry["apartment_type"] = "3Bed"

            # Acoustic properties
            if hasattr(ifc_elem, "IsDefinedBy"):
                for rel in ifc_elem.IsDefinedBy:
                    if rel.is_a("IfcRelDefinesByProperties"):
                        props = rel.RelatingPropertyDefinition
                        if props.is_a("IfcPropertySet"):
                            for prop in props.HasProperties:
                                if prop.is_a("IfcPropertySingleValue"):
                                    pname = prop.Name.lower()
                                    val = getattr(getattr(prop, "NominalValue", None), "wrappedValue", None)
                                    if "rt60" in pname:
                                        entry["RT60"] = float(val) if val else None
                                    elif "spl" in pname:
                                        entry["SPL"] = float(val) if val else None

            # IfcSlab: calculate surface area
            if ifc_elem.is_a() == "IfcSlab":
                try:
                    shape = ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(), ifc_elem).geometry
                    entry["area_m2"] = shape.Area()
                except:
                    pass

            result.append(entry)
        return result

    def save_model_data_to_db(self, db_path="sql/ifc_model_data.db"):
        data = self.extract_model_data()
        df = pd.DataFrame(data)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Save to SQLite
        conn = sqlite3.connect(db_path)
        df.to_sql('ifc_elements', conn, if_exists='replace', index=False)
        conn.close()
        
        print(f"OK Saved {len(data)} elements to {db_path}")
        return df

    def generate_heatmap_data(self, ifc_elem):
        # This would generate heatmap data for a specific element
        # For now, return dummy data
        return {
            "element_id": getattr(ifc_elem, "GlobalId", ""),
            "element_type": ifc_elem.is_a(),
            "acoustic_value": 0.5,  # Dummy value
            "coordinates": [0, 0, 0]  # Dummy coordinates
        }

    def load_ifc_file(self, path):
        """Load IFC file with enhanced rendering if enabled"""
        try:
            # Ensure display is properly initialized
            if not hasattr(self, '_display') or self._display is None:
                print("X Display not initialized. Initializing now...")
                self.InitDriver()
            
            # Wait a moment for display to be ready
            import time
            time.sleep(0.1)
            
            if self.enhanced_rendering_enabled:
                return self.load_ifc_with_enhanced_rendering(path)
            else:
                # Use simplified loading method
                return self._load_ifc_simple(path)
        except Exception as e:
            print(f"X Error in load_ifc_file: {e}")
            return False

    def _load_ifc_simple(self, path):
        """Simplified IFC loading method with minimal settings"""
        try:
            self.current_ifc_path = path
            ifc_file = ifcopenshell.open(path)
            
            # Clear existing shapes
            if hasattr(self, '_display') and self._display is not None:
                self._display.Context.RemoveAll(False)
            self.shape_to_ifc.clear()
            
            # Use only basic settings that are guaranteed to work
            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_PYTHON_OPENCASCADE, True)
            settings.set(settings.USE_WORLD_COORDS, True)
            
            # Define loadable element types
            loadable_types = [
                "IfcWall", "IfcSlab", "IfcBeam", "IfcColumn", "IfcDoor", "IfcWindow",
                "IfcRoof", "IfcStair", "IfcFurniture", "IfcSanitaryTerminal",
                "IfcRailing", "IfcCovering", "IfcPlate", "IfcMember", "IfcFooting",
                "IfcPile", "IfcBuildingElementProxy", "IfcDistributionElement",
                "IfcFlowTerminal", "IfcFlowSegment", "IfcFlowFitting", "IfcSpace"
            ]
            
            # Process each product
            products = ifc_file.by_type("IfcProduct")
            loaded_count = 0
            error_count = 0
            
            for product in products:
                try:
                    # Skip non-loadable element types
                    if product.is_a() not in loadable_types:
                        continue
                    
                    # Special handling for IfcSpace - create bounding box only if no representation exists
                    if product.is_a() == "IfcSpace":
                        # First try to create shape normally
                        if hasattr(product, "Representation") and product.Representation:
                            shape_result = ifcopenshell.geom.create_shape(settings, product)
                            # If normal shape creation fails, create bounding box
                            if not shape_result or not hasattr(shape_result, 'geometry') or shape_result.geometry is None:
                                shape_result = self.create_space_bounding_box(product, settings)
                        else:
                            # No representation, create bounding box
                            shape_result = self.create_space_bounding_box(product, settings)
                    else:
                        # Skip elements without representation
                        if not hasattr(product, "Representation") or not product.Representation:
                            continue
                        # Create shape with basic settings
                        shape_result = ifcopenshell.geom.create_shape(settings, product)
                    
                    if shape_result and hasattr(shape_result, 'geometry') and shape_result.geometry is not None:
                        # Create AIS shape with error handling
                        try:
                            ais_shape = AIS_Shape(shape_result.geometry)
                        except Exception as shape_error:
                            print(f"WARNING Failed to create AIS shape for {product.is_a()}: {str(shape_error)[:50]}...")
                            continue
                        
                        # Apply color
                        element_type = product.is_a()
                        color = self.color_map.get(element_type, self.default_color)
                        color_obj = Quantity_Color(color[0], color[1], color[2], Quantity_TOC_RGB)
                        self._display.Context.SetColor(ais_shape, color_obj, False)
                        
                        # For spaces, make them semi-transparent
                        if element_type == "IfcSpace":
                            self._display.Context.SetTransparency(ais_shape, 0.7, False)
                        
                        # Display the shape
                        self._display.Context.Display(ais_shape, False)
                        
                        # Store mapping
                        self.shape_to_ifc[ais_shape] = product
                        loaded_count += 1
                        
                        if loaded_count % 10 == 0:  # Progress indicator
                            print(f"PACKAGE Loaded {loaded_count} elements...")
                        
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:  # Only show first 5 errors to avoid spam
                        print(f"WARNING Error processing {product.is_a()}: {str(e)[:100]}...")
                    continue
            
            # Update display
            if hasattr(self, '_display') and self._display is not None:
                self._display.Context.UpdateCurrentViewer()
                self._display.FitAll()
            
            print(f"OK Loaded IFC: {loaded_count} elements (skipped {error_count} errors)")
            
            if loaded_count == 0:
                print("WARNING No geometry loaded. This might be due to:")
                print("   - IFC file has no geometric representation")
                print("   - Elements are not in supported types")
                print("   - Display driver issues")
                return False
            
            return True
            
        except Exception as e:
            print(f"X Error loading IFC: {e}")
            return False

    def generate_heatmap(self):
        # This would generate a heatmap visualization
        # For now, just print a message
        print("SEARCH Heatmap generation not implemented yet")
        return None

    def analyze_ifc_file(self, path):
        """Analyze IFC file and return element counts"""
        try:
            ifc_file = ifcopenshell.open(path)
            
            # Count elements by type
            element_counts = {}
            for product in ifc_file.by_type("IfcProduct"):
                elem_type = product.is_a()
                element_counts[elem_type] = element_counts.get(elem_type, 0) + 1
            
            return element_counts
            
        except Exception as e:
            print(f"X Error analyzing IFC file: {e}")
            return None

    def create_space_bounding_box(self, space, settings):
        """Create a bounding box representation for an IFC Space"""
        try:
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
            from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
            from OCC.Core.gp import gp_Trsf
            
            # Try to get space bounds from contained elements
            bounds = self.get_space_bounds_from_elements(space)
            
            if bounds:
                # Create a box based on the bounds
                min_x, min_y, min_z, max_x, max_y, max_z = bounds
                width = max_x - min_x
                height = max_y - min_y
                depth = max_z - min_z
                
                # Ensure minimum dimensions
                width = max(width, 1.0)
                height = max(height, 1.0)
                depth = max(depth, 1.0)
                
                # Create box
                box = BRepPrimAPI_MakeBox(gp_Pnt(min_x, min_y, min_z), width, height, depth).Shape()
                
                # Create a simple shape result object
                class SimpleShapeResult:
                    def __init__(self, geometry):
                        self.geometry = geometry
                
                return SimpleShapeResult(box)
            else:
                # Fallback: create a default box at origin
                default_box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 5.0, 5.0, 3.0).Shape()
                class SimpleShapeResult:
                    def __init__(self, geometry):
                        self.geometry = geometry
                return SimpleShapeResult(default_box)
                
        except Exception as e:
            print(f"WARNING Error creating space bounding box: {e}")
            return None
    
    def get_space_bounds_from_elements(self, space):
        """Get bounding box from elements contained in the space"""
        try:
            if not hasattr(space, 'GlobalId'):
                return None
                
            space_global_id = space.GlobalId
            
            # Find elements that are contained in this space
            min_x = min_y = min_z = float('inf')
            max_x = max_y = max_z = float('-inf')
            found_elements = False
            
            for ais_shape, ifc_elem in self.shape_to_ifc.items():
                # Check if this element is contained in the space
                if hasattr(ifc_elem, 'ContainedInStructure'):
                    for rel in ifc_elem.ContainedInStructure:
                        if (hasattr(rel, 'RelatingStructure') and 
                            hasattr(rel.RelatingStructure, 'GlobalId') and
                            rel.RelatingStructure.GlobalId == space_global_id):
                            
                            # Get element bounds
                            try:
                                shape = ais_shape.Shape()
                                if shape:
                                    bbox = shape.BoundingBox()
                                    if bbox:
                                        min_x = min(min_x, bbox.Xmin())
                                        min_y = min(min_y, bbox.Ymin())
                                        min_z = min(min_z, bbox.Zmin())
                                        max_x = max(max_x, bbox.Xmax())
                                        max_y = max(max_y, bbox.Ymax())
                                        max_z = max(max_z, bbox.Zmax())
                                        found_elements = True
                            except:
                                continue
            
            if found_elements:
                return (min_x, min_y, min_z, max_x, max_y, max_z)
            else:
                return None
                
        except Exception as e:
            print(f"WARNING Error getting space bounds: {e}")
            return None

class GraphViewer(QDockWidget):
    def __init__(self, parent=None):
        print("[DEBUG] GraphViewer.__init__() - Starting initialization")
        super().__init__("Graph Viewer", parent)
        print("[DEBUG] GraphViewer.__init__() - Super().__init__() completed")
        
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        print("[DEBUG] GraphViewer.__init__() - Set allowed areas")
        
        # Prevent the dock from being closed
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | 
                        QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        print("[DEBUG] GraphViewer.__init__() - Set dock features")
        
        # Create main widget to hold toolbar and web view
        main_widget = QWidget()
        self.setWidget(main_widget)
        print("[DEBUG] GraphViewer.__init__() - Created main widget")
        
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        print("[DEBUG] GraphViewer.__init__() - Created layout")
        
        # Create toolbar
        toolbar = QWidget()
        toolbar.setStyleSheet("background: #3C3C3C; padding: 5px; border-bottom: 1px solid #555;")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        print("[DEBUG] GraphViewer.__init__() - Created toolbar")
        
        # Graph selection dropdown
        graph_label = QLabel("Graph:")
        graph_label.setStyleSheet("color: #FDF6F6; font-weight: bold; margin-right: 5px;")
        toolbar_layout.addWidget(graph_label)
        print("[DEBUG] GraphViewer.__init__() - Added graph label")
        
        self.graph_combo = QComboBox()
        self.graph_combo.setStyleSheet("""
            QComboBox {
                background: #2C2C2C;
                color: #FDF6F6;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px 8px;
                min-width: 150px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #FDF6F6;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background: #2C2C2C;
                border: 1px solid #555;
                color: #FDF6F6;
                selection-background-color: #555;
            }
        """)
        toolbar_layout.addWidget(self.graph_combo)
        print("[DEBUG] GraphViewer.__init__() - Added graph combo")
        
        # Viewer type selector
        viewer_label = QLabel("Viewer:")
        viewer_label.setStyleSheet("color: #FDF6F6; font-weight: bold; margin-right: 5px;")
        toolbar_layout.addWidget(viewer_label)
        
        self.viewer_combo = QComboBox()
        self.viewer_combo.addItems(["Web Browser", "Text Browser", "Rich Text", "Simple HTML"])
        self.viewer_combo.setStyleSheet("""
            QComboBox {
                background: #2C2C2C;
                color: #FDF6F6;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px 8px;
                min-width: 120px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #FDF6F6;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background: #2C2C2C;
                border: 1px solid #555;
                color: #FDF6F6;
                selection-background-color: #555;
            }
        """)
        self.viewer_combo.currentTextChanged.connect(self.switch_viewer_type)
        toolbar_layout.addWidget(self.viewer_combo)
        print("[DEBUG] GraphViewer.__init__() - Added viewer type selector")
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #45a049;
            }
        """)
        refresh_btn.clicked.connect(self.refresh_view)
        toolbar_layout.addWidget(refresh_btn)
        print("[DEBUG] GraphViewer.__init__() - Added refresh button")
        
        # Open in browser button
        browser_btn = QPushButton("🌐 Open in Browser")
        browser_btn.setStyleSheet("""
            QPushButton {
                background: #2196F3;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1976D2;
            }
        """)
        browser_btn.clicked.connect(self.open_in_browser)
        toolbar_layout.addWidget(browser_btn)
        print("[DEBUG] GraphViewer.__init__() - Added browser button")
        
        toolbar_layout.addStretch()  # Add space to push controls to the left
        layout.addWidget(toolbar)
        print("[DEBUG] GraphViewer.__init__() - Added toolbar to layout")
        
        # Create stacked widget to hold different viewer types
        self.viewer_stack = QStackedWidget()
        layout.addWidget(self.viewer_stack)
        print("[DEBUG] GraphViewer.__init__() - Created viewer stack")
        
        # Initialize different viewer types
        self.init_viewers()
        
        # Populate graph dropdown and load default
        print("[DEBUG] GraphViewer.__init__() - Starting populate_graph_dropdown...")
        self.populate_graph_dropdown()
        
        # Connect dropdown change event
        try:
            self.graph_combo.currentTextChanged.connect(self.on_graph_selection_changed)
            print("[DEBUG] GraphViewer.__init__() - Connected dropdown change event")
        except Exception as e:
            print(f"[DEBUG] GraphViewer.__init__() - Error connecting dropdown event: {e}")
        
        print("[DEBUG] GraphViewer.__init__() - Initialization complete")
    
    def init_viewers(self):
        """Initialize different types of viewers"""
        print("[DEBUG] GraphViewer.init_viewers() - Starting...")

        # 1. Web Browser (QWebEngineView)
        try:
            print("[DEBUG] GraphViewer.init_viewers() - Creating QWebEngineView...")
            self.web_view = QWebEngineView()
            print("[DEBUG] GraphViewer.init_viewers() - QWebEngineView created successfully")

            # Enable JavaScript and other features
            settings = self.web_view.settings()
            try:
                settings.setAttribute(settings.WebAttribute.JavascriptEnabled, True)
            except AttributeError:
                try:
                    settings.setAttribute(settings.JavascriptEnabled, True)
                except Exception as e:
                    print(f"[DEBUG] - JavascriptEnabled attribute missing: {e}")

            for attr_name in ["LocalContentCanAccessRemoteUrls", "LocalContentCanAccessFileUrls", "AllowRunningInsecureContent"]:
                try:
                    attr = getattr(settings.WebAttribute, attr_name)
                except AttributeError:
                    try:
                        attr = getattr(settings, attr_name)
                    except AttributeError:
                        print(f"[DEBUG] - Attribute {attr_name} not found in WebAttribute or settings.")
                        continue
                try:
                    settings.setAttribute(attr, True)
                except Exception as e:
                    print(f"[DEBUG] - Could not set {attr_name}: {e}")

            print("[DEBUG] GraphViewer.init_viewers() - Web settings configured")

            try:
                self.web_view.page().javaScriptConsoleMessage.connect(self.handle_js_console)
                print("[DEBUG] GraphViewer.init_viewers() - JavaScript console connected")
            except Exception as e:
                print(f"[DEBUG] GraphViewer.init_viewers() - Error connecting JavaScript console: {e}")

        except Exception as e:
            print(f"[DEBUG] GraphViewer.init_viewers() - Error creating QWebEngineView: {e}")
            import traceback
            traceback.print_exc()
            # Create a fallback widget
            self.web_view = QLabel("QWebEngineView failed to create")
            self.web_view.setStyleSheet("background: #2C2C2C; color: #FF4444; padding: 20px;")

        self.viewer_stack.addWidget(self.web_view)
        print("[DEBUG] GraphViewer.init_viewers() - Added web view to stack")

        
        # 2. Text Browser (QTextBrowser)
        print("[DEBUG] GraphViewer.init_viewers() - Creating QTextBrowser...")
        self.text_browser = QTextBrowser()
        self.text_browser.setStyleSheet("""
            QTextBrowser {
                background: #2C2C2C;
                color: #FDF6F6;
                border: none;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
            }
        """)
        self.viewer_stack.addWidget(self.text_browser)
        print("[DEBUG] GraphViewer.init_viewers() - Added text browser to stack")
        
        # 3. Rich Text (QLabel with HTML)
        print("[DEBUG] GraphViewer.init_viewers() - Creating Rich Text Label...")
        self.rich_text_label = QLabel()
        self.rich_text_label.setStyleSheet("""
            QLabel {
                background: #2C2C2C;
                color: #FDF6F6;
                border: none;
                padding: 10px;
            }
        """)
        self.rich_text_label.setWordWrap(True)
        self.rich_text_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.viewer_stack.addWidget(self.rich_text_label)
        print("[DEBUG] GraphViewer.init_viewers() - Added rich text label to stack")
        
        # 4. Simple HTML (QTextEdit)
        print("[DEBUG] GraphViewer.init_viewers() - Creating Simple HTML TextEdit...")
        self.simple_html_edit = QTextEdit()
        self.simple_html_edit.setStyleSheet("""
            QTextEdit {
                background: #2C2C2C;
                color: #FDF6F6;
                border: none;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
            }
        """)
        self.simple_html_edit.setReadOnly(True)
        self.viewer_stack.addWidget(self.simple_html_edit)
        print("[DEBUG] GraphViewer.init_viewers() - Added simple HTML editor to stack")
        
        print("[DEBUG] GraphViewer.init_viewers() - All viewers initialized")
    
    def switch_viewer_type(self, viewer_type):
        """Switch between different viewer types"""
        print(f"[DEBUG] GraphViewer.switch_viewer_type() - Switching to: {viewer_type}")
        
        viewer_map = {
            "Web Browser": 0,
            "Text Browser": 1,
            "Rich Text": 2,
            "Simple HTML": 3
        }
        
        if viewer_type in viewer_map:
            index = viewer_map[viewer_type]
            self.viewer_stack.setCurrentIndex(index)
            print(f"[DEBUG] GraphViewer.switch_viewer_type() - Switched to index {index}")
            
            # Reload current content in new viewer
            current_graph = self.graph_combo.currentText()
            if current_graph and current_graph != "No graphs available":
                self.load_graph_file(current_graph)
    
    def handle_js_console(self, level, message, line, source):
        """Handle JavaScript console messages"""
        print(f"[GraphViewer JS] Level {level}: {message} (line {line}, source: {source})")
    
    def closeEvent(self, event):
        """Prevent the dock from being closed"""
        print("[DEBUG] GraphViewer.closeEvent() - Close event received, ignoring...")
        event.ignore()
        print("GraphViewer close event ignored - widget will stay open")
    
    def showEvent(self, event):
        """Handle show event"""
        print("[DEBUG] GraphViewer.showEvent() - Show event received")
        super().showEvent(event)
    
    def hideEvent(self, event):
        """Handle hide event"""
        print("[DEBUG] GraphViewer.hideEvent() - Hide event received")
        super().hideEvent(event)
    
    def populate_graph_dropdown(self):
        """Populate the dropdown with available graph files"""
        print("[DEBUG] GraphViewer.populate_graph_dropdown() - Starting...")
        try:
            available_graphs = self.get_available_graphs()
            print(f"[DEBUG] GraphViewer.populate_graph_dropdown() - Found {len(available_graphs)} graphs: {available_graphs}")
            
            self.graph_combo.clear()
            
            if available_graphs:
                self.graph_combo.addItems(available_graphs)
                print("[DEBUG] GraphViewer.populate_graph_dropdown() - Added items to combo")
                
                # Set default to BDG_IFC (1).html if available
                default_index = self.graph_combo.findText("BDG_IFC (1).html")
                if default_index >= 0:
                    self.graph_combo.setCurrentIndex(default_index)
                    print("[DEBUG] GraphViewer.populate_graph_dropdown() - Set default to BDG_IFC (1).html")
                else:
                    self.graph_combo.setCurrentIndex(0)
                    print("[DEBUG] GraphViewer.populate_graph_dropdown() - Set default to first item")
                
                print(f"OK Loaded {len(available_graphs)} graph files into dropdown")
                
                # Load the default graph immediately
                default_graph = self.graph_combo.currentText()
                print(f"[DEBUG] GraphViewer.populate_graph_dropdown() - Default graph: {default_graph}")
                
                if default_graph and default_graph != "No graphs available":
                    print("[DEBUG] GraphViewer.populate_graph_dropdown() - Loading default graph...")
                    self.load_graph_file(default_graph)
                else:
                    print("[DEBUG] GraphViewer.populate_graph_dropdown() - No valid default graph found")
            else:
                self.graph_combo.addItem("No graphs available")
                print("WARNING No graph files found")
        except Exception as e:
            print(f"[DEBUG] GraphViewer.populate_graph_dropdown() - Error: {e}")
            import traceback
            traceback.print_exc()
    
    def on_graph_selection_changed(self, selected_graph):
        """Handle graph selection change"""
        print(f"[DEBUG] GraphViewer.on_graph_selection_changed() - Selected: {selected_graph}")
        if selected_graph and selected_graph != "No graphs available":
            print("[DEBUG] GraphViewer.on_graph_selection_changed() - Loading selected graph...")
            self.load_graph_file(selected_graph)
    
    def load_graph_file(self, filename):
        """Load a specific graph HTML file with enhanced interactive support"""
        print(f"[DEBUG] GraphViewer.load_graph_file() - Loading: {filename}")
        
        # Use absolute path from project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        html_file_path = os.path.join(project_root, "models", "Graphs", filename)
        
        print(f"[DEBUG] GraphViewer.load_graph_file() - Full path: {html_file_path}")
        print(f"[DEBUG] GraphViewer.load_graph_file() - File exists: {os.path.exists(html_file_path)}")
        
        if os.path.exists(html_file_path):
            try:
                # Read the HTML content
                with open(html_file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                print(f"[DEBUG] GraphViewer.load_graph_file() - Read {len(html_content)} characters")
                
                # Check if this is an interactive graph
                is_interactive = False
                if "BDG_IFC" in filename:
                    # Only check for interactive content if it is a BDG_IFC file
                    if ("play" in html_content.lower() or "controls" in html_content.lower() or "interactive" in html_content.lower()):
                        is_interactive = True
                print(f"[DEBUG] GraphViewer.load_graph_file() - Is interactive: {is_interactive}")
                
                # Load content based on current viewer type
                current_viewer = self.viewer_combo.currentText()
                print(f"[DEBUG] GraphViewer.load_graph_file() - Current viewer: {current_viewer}")
                
                if current_viewer == "Web Browser":
                    if is_interactive:
                        # For interactive graphs, offer choice
                        self.offer_interactive_viewer_choice(filename, html_file_path, html_content)
                    else:
                        self.load_in_web_browser(html_file_path, html_content)
                elif current_viewer == "Text Browser":
                    self.load_in_text_browser(html_content)
                elif current_viewer == "Rich Text":
                    self.load_in_rich_text(html_content)
                elif current_viewer == "Simple HTML":
                    self.load_in_simple_html(html_content)
                
                print(f"OK Loaded graph visualization: {html_file_path}")
                return True
                
            except Exception as e:
                print(f"[DEBUG] GraphViewer.load_graph_file() - Error loading file: {e}")
                import traceback
                traceback.print_exc()
                self.show_fallback_content(filename, str(e))
                return False
        else:
            # Fallback to static content if file doesn't exist
            print(f"WARNING HTML file not found: {html_file_path}")
            self.show_fallback_content(filename, "File not found")
            return False
    
    def offer_interactive_viewer_choice(self, filename, file_path, html_content):
        """Offer user choice for interactive graph viewing"""
        print("[DEBUG] GraphViewer.offer_interactive_viewer_choice() - Offering choice...")
        
        from PyQt6.QtWidgets import QMessageBox
        
        msg = QMessageBox()
        msg.setWindowTitle("Interactive Graph Detected")
        msg.setText(f"'{filename}' appears to be an interactive graph with JavaScript controls.")
        msg.setInformativeText("How would you like to view it?")
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | 
            QMessageBox.StandardButton.No | 
            QMessageBox.StandardButton.Cancel
        )
        msg.button(QMessageBox.StandardButton.Yes).setText("Separate Window")
        msg.button(QMessageBox.StandardButton.No).setText("Embedded Viewer")
        msg.button(QMessageBox.StandardButton.Cancel).setText("External Browser")
        
        result = msg.exec()
        
        if result == QMessageBox.StandardButton.Yes:
            # Separate window
            self.load_interactive_graph_in_window(file_path, html_content)
        elif result == QMessageBox.StandardButton.No:
            # Embedded viewer
            self.load_interactive_graph(file_path, html_content)
        else:
            # External browser
            self.open_interactive_in_browser()
    
    def load_in_web_browser(self, file_path, html_content):
        """Load content in QWebEngineView with enhanced JavaScript support"""
        print("[DEBUG] GraphViewer.load_in_web_browser() - Loading in web browser...")
        try:
            # Check if this is an interactive graph (BDG_IFC)
            is_interactive = "BDG_IFC" in file_path or "interactive" in html_content.lower()
            
            if is_interactive:
                print("[DEBUG] GraphViewer.load_in_web_browser() - Detected interactive graph, using enhanced loading...")
                self.load_interactive_graph(file_path, html_content)
            else:
                # Try loading as file URL first
                file_url = QUrl.fromLocalFile(file_path)
                print(f"[DEBUG] GraphViewer.load_in_web_browser() - Loading URL: {file_url.toString()}")
                self.web_view.setUrl(file_url)
        except Exception as e:
            print(f"[DEBUG] GraphViewer.load_in_web_browser() - Error loading file URL: {e}")
            # Fallback: set HTML content directly
            try:
                self.web_view.setHtml(html_content, QUrl.fromLocalFile(file_path))
                print("[DEBUG] GraphViewer.load_in_web_browser() - Loaded HTML content directly")
            except Exception as e2:
                print(f"[DEBUG] GraphViewer.load_in_web_browser() - Error loading HTML content: {e2}")
    
    def load_interactive_graph(self, file_path, html_content):
        """Load interactive graphs with enhanced JavaScript support"""
        print("[DEBUG] GraphViewer.load_interactive_graph() - Loading interactive graph...")
        try:
            # Create a base URL for the file
            base_url = QUrl.fromLocalFile(os.path.dirname(file_path) + "/")
            print(f"[DEBUG] GraphViewer.load_interactive_graph() - Base URL: {base_url.toString()}")
            
            # Set enhanced web settings for interactive graphs
            settings = self.web_view.settings()
            try:
                settings.setAttribute(settings.WebAttribute.JavascriptEnabled, True)
            except AttributeError:
                settings.setAttribute(settings.JavascriptEnabled, True)
            try:
                settings.setAttribute(settings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            except AttributeError:
                settings.setAttribute(settings.LocalContentCanAccessRemoteUrls, True)
            try:
                settings.setAttribute(settings.WebAttribute.LocalContentCanAccessFileUrls, True)
            except AttributeError:
                settings.setAttribute(settings.LocalContentCanAccessFileUrls, True)
            settings.setAttribute(settings.AllowRunningInsecureContent, True)
            settings.setAttribute(settings.AllowWindowActivationFromJavaScript, True)
            settings.setAttribute(settings.JavascriptCanAccessClipboard, True)
            settings.setAttribute(settings.JavascriptCanOpenWindows, True)
            settings.setAttribute(settings.JavascriptCanPaste, True)
            settings.setAttribute(settings.LocalStorageEnabled, True)
            settings.setAttribute(settings.PersistentStorageEnabled, True)
            settings.setAttribute(settings.PluginsEnabled, True)
            settings.setAttribute(settings.WebGLEnabled, True)
            settings.setAttribute(settings.Accelerated2dCanvasEnabled, True)
            settings.setAttribute(settings.ScrollAnimatorEnabled, True)
            settings.setAttribute(settings.ErrorPageEnabled, True)
            settings.setAttribute(settings.FocusOnNavigationEnabled, True)
            settings.setAttribute(settings.FullScreenSupportEnabled, True)
            settings.setAttribute(settings.HyperlinkAuditingEnabled, True)
            settings.setAttribute(settings.LinksIncludedInFocusChain, True)
            try:
                settings.setAttribute(settings.WebAttribute.LocalContentCanAccessFileUrls, True)
            except AttributeError:
                settings.setAttribute(settings.LocalContentCanAccessFileUrls, True)
            try:
                settings.setAttribute(settings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            except AttributeError:
                settings.setAttribute(settings.LocalContentCanAccessRemoteUrls, True)
            settings.setAttribute(settings.SpatialNavigationEnabled, True)
            settings.setAttribute(settings.ScreenCaptureEnabled, True)
            try:
                settings.setAttribute(settings.WebAttribute.WebSecurityEnabled, False)
            except AttributeError:
                settings.setAttribute(settings.WebSecurityEnabled, False)  # For local files
            
            print("[DEBUG] GraphViewer.load_interactive_graph() - Enhanced settings applied")
            
            # Load the HTML content with the base URL
            self.web_view.setHtml(html_content, base_url)
            print("[DEBUG] GraphViewer.load_interactive_graph() - Interactive graph loaded with base URL")
            
            # Add JavaScript console message handling
            try:
                self.web_view.page().javaScriptConsoleMessage.connect(self.handle_js_console)
                print("[DEBUG] GraphViewer.load_interactive_graph() - JavaScript console connected")
            except Exception as e:
                print(f"[DEBUG] GraphViewer.load_interactive_graph() - Error connecting JS console: {e}")
            
        except Exception as e:
            print(f"[DEBUG] GraphViewer.load_interactive_graph() - Error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to regular loading
            self.web_view.setHtml(html_content, QUrl.fromLocalFile(file_path))
    
    def create_interactive_graph_viewer(self):
        """Create a specialized viewer for interactive graphs"""
        print("[DEBUG] GraphViewer.create_interactive_graph_viewer() - Creating specialized viewer...")
        
        # Create a new window for interactive graphs
        from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel
        from PyQt6.QtCore import Qt
        
        self.interactive_window = QMainWindow()
        self.interactive_window.setWindowTitle("Interactive Graph Viewer")
        self.interactive_window.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.interactive_window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create toolbar
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        
        # Add controls
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_interactive_view)
        toolbar_layout.addWidget(refresh_btn)
        
        open_browser_btn = QPushButton("🌐 Open in External Browser")
        open_browser_btn.clicked.connect(self.open_interactive_in_browser)
        toolbar_layout.addWidget(open_browser_btn)
        
        close_btn = QPushButton("X Close")
        close_btn.clicked.connect(self.interactive_window.close)
        toolbar_layout.addWidget(close_btn)
        
        toolbar_layout.addStretch()
        layout.addWidget(toolbar)
        
        # Create web view for interactive content
        try:
            self.interactive_web_view = QWebEngineView()
            
            # Apply enhanced settings
            settings = self.interactive_web_view.settings()
            try:
                settings.setAttribute(settings.WebAttribute.JavascriptEnabled, True)
            except AttributeError:
                settings.setAttribute(settings.JavascriptEnabled, True)
            try:
                settings.setAttribute(settings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            except AttributeError:
                settings.setAttribute(settings.LocalContentCanAccessRemoteUrls, True)
            try:
                settings.setAttribute(settings.WebAttribute.LocalContentCanAccessFileUrls, True)
            except AttributeError:
                settings.setAttribute(settings.LocalContentCanAccessFileUrls, True)
            settings.setAttribute(settings.AllowRunningInsecureContent, True)
            try:
                settings.setAttribute(settings.WebAttribute.WebSecurityEnabled, False)
            except AttributeError:
                settings.setAttribute(settings.WebSecurityEnabled, False)
            settings.setAttribute(settings.WebGLEnabled, True)
            settings.setAttribute(settings.Accelerated2dCanvasEnabled, True)
            
            # Connect JavaScript console
            self.interactive_web_view.page().javaScriptConsoleMessage.connect(self.handle_js_console)
            
            layout.addWidget(self.interactive_web_view)
            print("[DEBUG] GraphViewer.create_interactive_graph_viewer() - Interactive viewer created")
            
        except Exception as e:
            print(f"[DEBUG] GraphViewer.create_interactive_graph_viewer() - Error creating web view: {e}")
            # Fallback: show message
            error_label = QLabel("Interactive graph viewer could not be created. Use 'Open in External Browser' instead.")
            error_label.setStyleSheet("color: red; padding: 20px;")
            layout.addWidget(error_label)
    
    def load_interactive_graph_in_window(self, file_path, html_content):
        """Load interactive graph in a separate window"""
        print("[DEBUG] GraphViewer.load_interactive_graph_in_window() - Loading in separate window...")
        
        if not hasattr(self, 'interactive_window'):
            self.create_interactive_graph_viewer()
        
        try:
            # Create base URL
            base_url = QUrl.fromLocalFile(os.path.dirname(file_path) + "/")
            
            # Load content
            self.interactive_web_view.setHtml(html_content, base_url)
            
            # Show window
            self.interactive_window.show()
            self.interactive_window.raise_()
            
            print("[DEBUG] GraphViewer.load_interactive_graph_in_window() - Interactive window shown")
            
        except Exception as e:
            print(f"[DEBUG] GraphViewer.load_interactive_graph_in_window() - Error: {e}")
            # Fallback to external browser
            self.open_interactive_in_browser()
    
    def refresh_interactive_view(self):
        """Refresh the interactive graph view"""
        print("[DEBUG] GraphViewer.refresh_interactive_view() - Refreshing...")
        try:
            if hasattr(self, 'interactive_web_view'):
                self.interactive_web_view.reload()
                print("[DEBUG] GraphViewer.refresh_interactive_view() - View refreshed")
        except Exception as e:
            print(f"[DEBUG] GraphViewer.refresh_interactive_view() - Error: {e}")
    
    def open_interactive_in_browser(self):
        """Open interactive graph in external browser"""
        print("[DEBUG] GraphViewer.open_interactive_in_browser() - Opening in external browser...")
        current_graph = self.graph_combo.currentText()
        
        if current_graph and current_graph != "No graphs available":
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            html_file_path = os.path.join(project_root, "models", "Graphs", current_graph)
            
            if os.path.exists(html_file_path):
                try:
                    import webbrowser
                    # Use file:// protocol for local files
                    file_url = f"file:///{html_file_path.replace(os.sep, '/')}"
                    webbrowser.open(file_url)
                    print(f"OK Opened {current_graph} in external browser")
                except Exception as e:
                    print(f"[DEBUG] GraphViewer.open_interactive_in_browser() - Error: {e}")
            else:
                print(f"WARNING File not found: {html_file_path}")
    
    def load_in_text_browser(self, html_content):
        """Load content in QTextBrowser"""
        print("[DEBUG] GraphViewer.load_in_text_browser() - Loading in text browser...")
        try:
            # Convert HTML to plain text for better display
            import re
            # Remove HTML tags
            text_content = re.sub(r'<[^>]+>', '', html_content)
            # Decode HTML entities
            import html
            text_content = html.unescape(text_content)
            # Clean up whitespace
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            self.text_browser.setPlainText(text_content)
            print("[DEBUG] GraphViewer.load_in_text_browser() - Loaded text content")
        except Exception as e:
            print(f"[DEBUG] GraphViewer.load_in_text_browser() - Error: {e}")
            self.text_browser.setPlainText(f"Error loading content: {str(e)}")
    
    def load_in_rich_text(self, html_content):
        """Load content in QLabel with rich text"""
        print("[DEBUG] GraphViewer.load_in_rich_text() - Loading in rich text label...")
        try:
            # Extract title and basic info from HTML
            import re
            title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE)
            title = title_match.group(1) if title_match else "Graph Visualization"
            
            # Create a simplified HTML display
            simplified_html = f"""
            <html>
            <body style="color: #FDF6F6; font-family: Arial, sans-serif;">
                <h2 style="color: #FF9500;">{title}</h2>
                <p><strong>File loaded successfully!</strong></p>
                <p>Content length: {len(html_content)} characters</p>
                <hr>
                <p style="font-size: 10px; color: #888;">
                    <em>Use "Open in Browser" button to view the full interactive graph.</em>
                </p>
            </body>
            </html>
            """
            
            self.rich_text_label.setText(simplified_html)
            print("[DEBUG] GraphViewer.load_in_rich_text() - Loaded rich text content")
        except Exception as e:
            print(f"[DEBUG] GraphViewer.load_in_rich_text() - Error: {e}")
            self.rich_text_label.setText(f"Error loading content: {str(e)}")
    
    def load_in_simple_html(self, html_content):
        """Load content in QTextEdit with HTML support"""
        print("[DEBUG] GraphViewer.load_in_simple_html() - Loading in simple HTML editor...")
        try:
            # Show HTML content with syntax highlighting
            self.simple_html_edit.setPlainText(html_content)
            print("[DEBUG] GraphViewer.load_in_simple_html() - Loaded HTML content")
        except Exception as e:
            print(f"[DEBUG] GraphViewer.load_in_simple_html() - Error: {e}")
            self.simple_html_edit.setPlainText(f"Error loading content: {str(e)}")
    
    def open_in_browser(self):
        """Open current graph in default browser"""
        print("[DEBUG] GraphViewer.open_in_browser() - Starting...")
        current_graph = self.graph_combo.currentText()
        print(f"[DEBUG] GraphViewer.open_in_browser() - Current graph: {current_graph}")
        
        if current_graph and current_graph != "No graphs available":
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            html_file_path = os.path.join(project_root, "models", "Graphs", current_graph)
            print(f"[DEBUG] GraphViewer.open_in_browser() - Full path: {html_file_path}")
            
            if os.path.exists(html_file_path):
                try:
                    import webbrowser
                    webbrowser.open(f"file://{html_file_path}")
                    print(f"OK Opened {current_graph} in browser")
                except Exception as e:
                    print(f"[DEBUG] GraphViewer.open_in_browser() - Error opening browser: {e}")
            else:
                print(f"WARNING File not found: {html_file_path}")
    
    def show_fallback_content(self, missing_file, error_msg=""):
        """Show fallback content when graph file is not found"""
        print(f"[DEBUG] GraphViewer.show_fallback_content() - Missing: {missing_file}, Error: {error_msg}")
        
        fallback_html = f"""
        <html>
        <body style="color: #FDF6F6; font-family: Arial, sans-serif; background: #2C2C2C; padding: 20px;">
            <h2 style="color: #FF4444;">WARNING Graph Loading Issue</h2>
            <p><strong>Could not load:</strong> {missing_file}</p>
            <p><strong>Error:</strong> {error_msg}</p>
            <p><strong>Expected location:</strong> <code>models/Graphs/{missing_file}</code></p>
            <hr>
            <h3 style="color: #FF9500;">Available Viewer Options:</h3>
            <ul>
                <li><strong>Web Browser:</strong> Full interactive HTML (requires QWebEngine)</li>
                <li><strong>Text Browser:</strong> Plain text view of content</li>
                <li><strong>Rich Text:</strong> Simplified HTML display</li>
                <li><strong>Simple HTML:</strong> Raw HTML code view</li>
            </ul>
            <p><em>Try switching viewer types or use "Open in Browser" button.</em></p>
        </body>
        </html>
        """
        
        # Show fallback in all viewers
        try:
            self.web_view.setHtml(fallback_html)
            self.text_browser.setPlainText(f"Graph Loading Issue\n\nCould not load: {missing_file}\nError: {error_msg}")
            self.rich_text_label.setText(fallback_html)
            self.simple_html_edit.setPlainText(fallback_html)
            print("[DEBUG] GraphViewer.show_fallback_content() - Fallback content set")
        except Exception as e:
            print(f"[DEBUG] GraphViewer.show_fallback_content() - Error setting fallback: {e}")
    
    def refresh_view(self):
        """Refresh the current graph view"""
        print("[DEBUG] GraphViewer.refresh_view() - Starting refresh...")
        try:
            current_graph = self.graph_combo.currentText()
            if current_graph and current_graph != "No graphs available":
                self.load_graph_file(current_graph)
                print("OK Graph view refreshed")
            else:
                print("WARNING No graph selected to refresh")
        except Exception as e:
            print(f"[DEBUG] GraphViewer.refresh_view() - Error refreshing: {e}")
    
    def get_available_graphs(self):
        """Get list of available graph files"""
        print("[DEBUG] GraphViewer.get_available_graphs() - Starting...")
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            graphs_dir = os.path.join(project_root, "models", "Graphs")
            print(f"[DEBUG] GraphViewer.get_available_graphs() - Looking in: {graphs_dir}")
            print(f"[DEBUG] GraphViewer.get_available_graphs() - Directory exists: {os.path.exists(graphs_dir)}")
            
            if os.path.exists(graphs_dir):
                html_files = [f for f in os.listdir(graphs_dir) if f.endswith('.html')]
                print(f"[DEBUG] GraphViewer.get_available_graphs() - Found HTML files: {html_files}")
                return html_files
            else:
                print("[DEBUG] GraphViewer.get_available_graphs() - Graphs directory does not exist")
                return []
        except Exception as e:
            print(f"[DEBUG] GraphViewer.get_available_graphs() - Error: {e}")
            import traceback
            traceback.print_exc()
            return []

class EcoformMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from scripts.core.neo4j_interface import Neo4jConnector
        self.neo4j = Neo4jConnector(password="123456789", lazy_init=True)  # 🔑 Replace with your actual password
        self.setWindowTitle("Ecoform Acoustic Copilot")
        self.resize(1440, 820)
        self.central = QWidget()
        self.setCentralWidget(self.central)
        self.setStyleSheet(f"background: {COLOR_MAIN_BG};")
        main_layout = QHBoxLayout(self.central)

        # -- LEFT PANEL --
        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setSpacing(16)
        left_layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Ecoform Acoustic Copilot")
        title.setFont(FONT_TITLE)
        title.setStyleSheet(f"background: {COLOR_TITLE_BG}; color: {COLOR_TITLE_TEXT}; border-radius: 12px; padding: 20px; border: 2px solid {COLOR_TAB_BG};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(title)

        # --- Tabs for Inputs ---
        tabs = QTabWidget()
        geo_tab = QWidget()
        tab_style = f"""
        QTabWidget::pane {{ border: none; }}
        QTabBar::tab {{
            font-weight: bold;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 13px;
            margin: 2px;
            background: {COLOR_TAB_BG};
            color: {COLOR_TAB_TEXT};
            font-weight: bold;
            padding: 8px 20px;
            border-radius: 10px;
        }}
        QTabBar::tab:selected {{ background: {COLOR_DROPDOWN_FRAME}; }}
    """

        
        tabs.setStyleSheet(tab_style)


        # --- Geometry Inputs Tab ---
        geo_tab = QWidget()
        geo_layout = QVBoxLayout(geo_tab)
        geo_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        geo_layout.addWidget(QLabel("Apartment Type Group:", font=FONT_LABEL))
        self.geo_apartment = QComboBox()
        self.geo_apartment.addItems(["1Bed", "2Bed", "3Bed"])
        self.geo_apartment.setStyleSheet(f"background: {COLOR_DROPDOWN_BG}; color: {COLOR_INPUT_TEXT}; border: 2px solid {COLOR_DROPDOWN_FRAME}; border-radius: 6px; padding: 10px; font-size: 13px;")
        geo_layout.addWidget(self.geo_apartment)

        geo_layout.addWidget(QLabel("Wall Material:", font=FONT_LABEL))
        self.geo_wall = QComboBox()
        self.geo_wall.addItems([
            "Painted Brick", "Unpainted Brick", "Concrete Block (Coarse)", "Concrete Block (Painted)",
            "Gypsum Board", "Plaster on Masonry", "Plaster with Wallpaper Backing", "Wood Paneling",
            "Acoustic Plaster", "Fiberglass Board"])
        self.geo_wall.setStyleSheet(f"background: {COLOR_DROPDOWN_BG}; color: {COLOR_INPUT_TEXT}; border: 2px solid {COLOR_DROPDOWN_FRAME}; border-radius: 6px; padding: 10px; font-size: 13px;")
        geo_layout.addWidget(self.geo_wall)

        geo_layout.addWidget(QLabel("Window Material:", font=FONT_LABEL))
        self.geo_window = QComboBox()
        self.geo_window.addItems([
            "Single Pane Glass", "Double Pane Glass", "Laminated Glass", "Wired Glass", "Frosted Glass",
            "Insulated Glazing Unit", "Glass Block", "Glazed Ceramic Tile", "Large Pane Glass", "Small Pane Glass"])
        self.geo_window.setStyleSheet(f"background: {COLOR_DROPDOWN_BG}; color: {COLOR_INPUT_TEXT}; border: 2px solid {COLOR_DROPDOWN_FRAME}; border-radius: 6px; padding: 10px; font-size: 13px;")
        geo_layout.addWidget(self.geo_window)

        geo_layout.addWidget(QLabel("Floor Material:", font=FONT_LABEL))
        self.geo_floor = QComboBox()
        self.geo_floor.addItems([
            "Marble", "Terrazzo", "Vinyl Tile", "Wood Parquet", "Wood Flooring on Joists",
            "Thin Carpet on Concrete", "Thin Carpet on Wood", "Medium Pile Carpet", "Thick Pile Carpet", "Cork Floor Tiles"])
        self.geo_floor.setStyleSheet(f"background: {COLOR_DROPDOWN_BG}; color: {COLOR_INPUT_TEXT}; border: 2px solid {COLOR_DROPDOWN_FRAME}; border-radius: 6px; padding: 10px; font-size: 13px;")
        geo_layout.addWidget(self.geo_floor)
        tabs.addTab(geo_tab, "Geometry Inputs")

        # --- Scenario Inputs Tab ---
        scenario_tab = QWidget()
        scenario_layout = QVBoxLayout(scenario_tab)
        scenario_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scenario_layout.addWidget(QLabel("Zone Sound Scenario:", font=FONT_LABEL))
        self.scenario_box = QComboBox()
        self.scenario_box.addItems([
            "High density + High traffic", "High density + Medium traffic", "High density + Light traffic",
            "Medium density + High traffic", "Medium density + Medium traffic", "Medium density + Light traffic",
            "Low density + High traffic", "Low density + Medium traffic", "Low density + Light traffic"])
        self.scenario_box.setStyleSheet(f"background: {COLOR_DROPDOWN_BG}; color: {COLOR_INPUT_TEXT}; border: 2px solid {COLOR_DROPDOWN_FRAME}; border-radius: 6px; padding: 10px; font-size: 13px;")
        scenario_layout.addWidget(self.scenario_box)

        scenario_layout.addWidget(QLabel("Zone:", font=FONT_LABEL))
        self.zone_box = QComboBox()
        self.zone_box.addItems([
            "HD-Urban-V1", "MD-Urban-V2", "LD-Urban-V3", "Ind-Zone-V0",
            "Roadside-V1", "Roadside-V2", "Roadside-V3", "GreenEdge-V3"])
        self.zone_box.setStyleSheet(f"background: {COLOR_DROPDOWN_BG}; color: {COLOR_INPUT_TEXT}; border: 2px solid {COLOR_DROPDOWN_FRAME}; border-radius: 6px; padding: 10px; font-size: 13px;")
        scenario_layout.addWidget(self.zone_box)

        scenario_layout.addWidget(QLabel("Activity:", font=FONT_LABEL))
        self.activity_box = QComboBox()
        self.activity_box.addItems([
            "Sleeping", "Working", "Living", "Dining", "Learning", "Healing", "Exercise", "Co-working"])
        self.activity_box.setStyleSheet(f"background: {COLOR_DROPDOWN_BG}; color: {COLOR_INPUT_TEXT}; border: 2px solid {COLOR_DROPDOWN_FRAME}; border-radius: 6px; padding: 10px; font-size: 13px;")
        scenario_layout.addWidget(self.activity_box)

        scenario_layout.addWidget(QLabel("Time Period:", font=FONT_LABEL))
        self.time_box = QComboBox()
        self.time_box.addItems(["Day", "Night"])
        self.time_box.setStyleSheet(f"background: {COLOR_DROPDOWN_BG}; color: {COLOR_INPUT_TEXT}; border: 2px solid {COLOR_DROPDOWN_FRAME}; border-radius: 6px; padding: 10px; font-size: 13px;")
        scenario_layout.addWidget(self.time_box)

        scenario_layout.addWidget(QLabel("Custom Sound Upload (WAV):", font=FONT_LABEL))
        self.upload_btn = QPushButton("Upload WAV File")
        self.upload_btn.setStyleSheet(f"background: {COLOR_INPUT_BG}; color: {COLOR_INPUT_TEXT}; font-weight: bold; padding: 12px 20px; border-radius: 8px; border: none; font-size: 13px;")
        scenario_layout.addWidget(self.upload_btn)

        tabs.addTab(scenario_tab, "Scenario Inputs")
        left_layout.addWidget(tabs)

        # Preloaded models dropdown
        preloaded_label = QLabel("Preloaded Models:")
        preloaded_label.setFont(FONT_LABEL)
        preloaded_label.setStyleSheet(f"color: {COLOR_INPUT_TEXT}; font-weight: bold; margin-bottom: 5px;")
        left_layout.addWidget(preloaded_label)
        
        self.preloaded_combo = QComboBox()
        self.preloaded_combo.addItem("Select a preloaded model...")
        self.preloaded_combo.setStyleSheet(f"""
            QComboBox {{
                background: {COLOR_DROPDOWN_BG};
                color: {COLOR_INPUT_TEXT};
                border: 2px solid {COLOR_DROPDOWN_FRAME};
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
                min-width: 200px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {COLOR_DROPDOWN_FRAME};
                margin-right: 5px;
            }}
            QComboBox QAbstractItemView {{
                background: {COLOR_DROPDOWN_BG};
                border: 2px solid {COLOR_DROPDOWN_FRAME};
                color: {COLOR_INPUT_TEXT};
                selection-background-color: {COLOR_DROPDOWN_FRAME};
            }}
        """)
        self.preloaded_combo.currentTextChanged.connect(self.on_preloaded_model_selected)
        left_layout.addWidget(self.preloaded_combo)
        
        # Load preloaded models
        self.load_preloaded_models()

        self.threeD_btn = QPushButton("3D File Upload")
        self.threeD_btn.setStyleSheet(f"background: {COLOR_INPUT_BG}; color: {COLOR_INPUT_TEXT}; font-weight: bold; padding: 12px 20px; border-radius: 8px; border: none; font-size: 13px;")
        self.threeD_btn.clicked.connect(self.handle_ifc_upload)
        left_layout.addWidget(self.threeD_btn)

        # Add refresh button for IFC data
        self.refresh_btn = QPushButton("🔄 Refresh IFC Data")
        self.refresh_btn.setStyleSheet(f"background: {COLOR_INPUT_BG}; color: {COLOR_INPUT_TEXT}; font-weight: bold; padding: 12px 20px; border-radius: 8px; border: none; font-size: 13px;")
        self.refresh_btn.clicked.connect(self.force_refresh_ifc_data)
        self.refresh_btn.setEnabled(False)  # Disabled until IFC is loaded
        left_layout.addWidget(self.refresh_btn)

        self.eval_btn = QPushButton("Evaluate Scenario")
        self.eval_btn.setStyleSheet(f"background: {COLOR_INPUT_BG}; color: {COLOR_INPUT_TEXT}; font-weight: bold; font-size: 16px; border-radius: 8px;")
        left_layout.addWidget(self.eval_btn)
        left_layout.addStretch(1)
        main_layout.addWidget(self.left_panel)

        # -- RIGHT PANEL: all widgets created only once! --
        # Chatbot panel
        self.chat_panel = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_panel)
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setFont(FONT_CHAT)  # Apply larger chat font
        self.chat_history.setStyleSheet(f"background: {COLOR_CHAT_BG}; color: {COLOR_CHAT_TEXT}; font-weight: normal; border-radius: 8px; padding: 12px; border: 1px solid #444;")
        # Make chat box responsive - remove fixed height constraints
        self.chat_history.setMinimumHeight(200)  # Minimum height
        self.chat_history.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Allow expansion
        self.chat_layout.addWidget(self.chat_history, stretch=1)  # Give it more space
        self.chat_input = QLineEdit()
        self.chat_input.setStyleSheet(f"background: {COLOR_DROPDOWN_BG}; color: {COLOR_CHAT_TEXT}; border: 2px solid {COLOR_DROPDOWN_FRAME}; border-radius: 6px; padding: 10px; font-size: 13px;")
        self.chat_input.setPlaceholderText("Type your question...")
        self.chat_input.setFont(FONT_CHAT)  # Apply larger chat font
        self.chat_layout.addWidget(self.chat_input)
        self.send_btn = QPushButton("Send")
        self.send_btn.setStyleSheet(f"background: {COLOR_INPUT_BG}; color: {COLOR_INPUT_TEXT}; font-weight: bold; padding: 12px 20px; border-radius: 8px; border: none; font-size: 13px;")
        self.send_btn.clicked.connect(self.on_chatbot_send)
        self.chat_layout.addWidget(self.send_btn)

        # Right panel layout
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(8, 8, 8, 8)

        # 3D Viewer
        self.viewer = MyViewer(self)
        self.viewer.InitDriver()

        # Create the dock widget and connect it to the viewer
        self.info_dock = EnhancedInfoDock(self)
        self.viewer.info_dock = self.info_dock  # Connect viewer to info panel
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.info_dock)

        right_layout.addWidget(self.chat_panel)
        right_layout.addWidget(self.viewer, stretch=1)

        # Compact Action buttons row
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)  # Reduce spacing between buttons
        
        # Main action buttons (smaller)
        main_buttons = [
            ("Show Acoustic Analysis", self.show_acoustic_analysis_with_overlay),  # Combined function
            ("Capture Screen", self.capture_screen),  # Updated to use capture function
            ("Upload Neo4j", self.upload_to_neo4j),
            ("Export", self.export_ifc_to_excel)
        ]
        
        for name, callback in main_buttons:
            btn = QPushButton(name)
            btn.setStyleSheet(small_btn_style)  # Use smaller button style
            if callback:
                btn.clicked.connect(callback)
            btn_layout.addWidget(btn)

        right_layout.addLayout(btn_layout)

        # Compact Enhanced Rendering Controls (hidden in Dev Tools)
        enhanced_group = QGroupBox("Dev Tools")
        enhanced_group.setStyleSheet(f"""
            QGroupBox {{
                background: {COLOR_DROPDOWN_BG};
                border: 2px solid #666;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
                color: {COLOR_INPUT_TEXT};
                font-weight: bold;
                font-size: 13px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
            }}
        """)
        enhanced_group.setVisible(False)  # Hide the dev tools
        enhanced_layout = QVBoxLayout(enhanced_group)
        enhanced_layout.setSpacing(4)  # Reduce spacing
        
        # Enhanced rendering toggle
        self.enhanced_btn = QPushButton("Enable Enhanced Rendering")
        self.enhanced_btn.setStyleSheet(small_btn_style)
        self.enhanced_btn.clicked.connect(self.toggle_enhanced_rendering)
        enhanced_layout.addWidget(self.enhanced_btn)
        
        # Reset colors button
        self.reset_btn = QPushButton("Reset to Element Colors")
        self.reset_btn.setStyleSheet(small_btn_style)
        self.reset_btn.clicked.connect(self.reset_element_colors)
        self.reset_btn.setEnabled(False)  # Disabled until IFC is loaded
        enhanced_layout.addWidget(self.reset_btn)
        
        # Debug buttons (very small)
        debug_layout = QHBoxLayout()
        debug_layout.setSpacing(2)
        
        debug_buttons = [
            ("Debug ML", self.debug_ml_integration),
            ("Debug IFC", self.debug_ifc_analysis),
            ("Debug Acoustic", self.debug_acoustic_analysis),
            ("Analyze LVL4_2B_39", lambda: self.analyze_specific_space_ui("LVL4_2B_39")),
            ("Reset Acoustic", self.reset_acoustic_failure_overlay)
        ]
        
        for name, callback in debug_buttons:
            btn = QPushButton(name)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: #4A4A4A;
                    color: {TEXT_MAIN};
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: bold;
                    font-size: 10px;
                }}
                QPushButton:hover {{
                    background: #666666;
                }}
            """)
            btn.clicked.connect(callback)
            debug_layout.addWidget(btn)
        
        enhanced_layout.addLayout(debug_layout)
        right_layout.addWidget(enhanced_group)

        # Return to Inputs button
        self.return_to_inputs_btn = QPushButton("Return to Inputs")
        self.return_to_inputs_btn.setStyleSheet(f"background: {COLOR_INPUT_BG}; color: {COLOR_INPUT_TEXT}; font-weight: bold; padding: 12px 20px; border-radius: 8px; border: none; font-size: 13px;")
        self.return_to_inputs_btn.clicked.connect(self.show_inputs_panel)
        right_layout.addWidget(self.return_to_inputs_btn)

        main_layout.addWidget(right_panel)

        # --- BUTTON CONNECTIONS ---
        self.eval_btn.clicked.connect(self.on_evaluate_clicked)

        # --- OCC Selection Polling ---
        self.last_selected = None
        self.selection_timer = QTimer(self)
        self.selection_timer.timeout.connect(self.check_occ_selection)
        self.selection_timer.start(200)

        self.left_panel.setStyleSheet(panel_style)
        self.eval_btn.setStyleSheet(btn_style)
        self.threeD_btn.setStyleSheet(btn_style)
        self.geo_apartment.setStyleSheet(input_style)
        self.geo_wall.setStyleSheet(input_style)
        self.geo_window.setStyleSheet(input_style)
        self.geo_floor.setStyleSheet(input_style)
        self.chat_history.setStyleSheet(chat_style)
        self.chat_input.setStyleSheet(chat_style)
        tabs.setStyleSheet("""
            QTabWidget::pane { border: none; }
            QTabBar::tab {
                font-weight: bold;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 13px;
                margin: 2px;
                background: #eee;
                color: #333;
                font-weight: bold; padding: 8px 20px; border-radius: 10px;
            }
            QTabBar::tab:selected { background: #ccc; }
        """)


        # ... existing code ...
        self.setup_analyze_all_checkbox()

        # Add menu items
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        analysis_menu = menubar.addMenu('Analysis')
        tools_menu = menubar.addMenu('Tools')
        
        # File menu actions
        upload_action = QAction('Upload IFC', self)
        upload_action.triggered.connect(self.handle_ifc_upload)
        file_menu.addAction(upload_action)
        
        # Analysis menu actions
        acoustic_action = QAction('Acoustic Analysis', self)
        acoustic_action.triggered.connect(self.show_acoustic_analysis_with_overlay)  # Updated to combined function
        analysis_menu.addAction(acoustic_action)
        
        # Tools menu actions
        graph_action = QAction('Graph Viewer', self)
        print("[DEBUG] Creating Graph Viewer menu action...")
        graph_action.triggered.connect(self.show_graph_viewer)
        print("[DEBUG] Graph Viewer menu action connected to show_graph_viewer")
        tools_menu.addAction(graph_action)
        print("[DEBUG] Graph Viewer menu action added to tools menu")
        
        capture_action = QAction('Capture Screen', self)
        capture_action.triggered.connect(self.capture_screen)
        tools_menu.addAction(capture_action)
        
        neo4j_action = QAction('Upload to Neo4j', self)
        neo4j_action.triggered.connect(self.upload_to_neo4j)
        tools_menu.addAction(neo4j_action)
        
        export_action = QAction('Export to Excel', self)
        export_action.triggered.connect(self.export_ifc_to_excel)
        tools_menu.addAction(export_action)

    def setup_analyze_all_checkbox(self):
        from PyQt6.QtWidgets import QCheckBox
        self.analyze_all_checkbox = QCheckBox("Analyze ALL spaces (may be slow)")
        self.analyze_all_checkbox.setChecked(False)
        self.left_panel.layout().addWidget(self.analyze_all_checkbox)

    def run_acoustic_failure_analysis(self):
        """Run acoustic failure analysis and display results in chat"""
        try:
            # Check if IFC file is loaded
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                self.chat_history.append("<span style='color:red;'>X No IFC file loaded. Please load an IFC file first.</span>")
                return
            # Show analysis starting message
            self.chat_history.append("<span style='color:#FF9500;'><b>SEARCH Starting Acoustic Failure Analysis...</b></span>")
            # Determine if analyze_all is checked
            analyze_all = False
            if hasattr(self, 'analyze_all_checkbox'):
                analyze_all = self.analyze_all_checkbox.isChecked()
            # Run the analysis with no timeout
            import threading
            result = None
            analysis_complete = threading.Event()
            def run_analysis():
                nonlocal result
                try:
                    result = self.identify_failing_acoustic_spaces(analyze_all=analyze_all)
                    analysis_complete.set()
                except Exception as e:
                    result = f"Analysis error: {str(e)}"
                    analysis_complete.set()
            analysis_thread = threading.Thread(target=run_analysis)
            analysis_thread.daemon = True
            analysis_thread.start()
            # Wait for analysis to complete (no timeout)
            analysis_complete.wait()
            if result:
                # Display results
                self.chat_history.append(f"<span style='color:#FF9500;'><b>SEARCH Acoustic Failure Analysis Results:</b></span>")
                self.chat_history.append(f"<span style='color:#FDF6F6;'>{result}</span>")
                # Create visual overlay for failing spaces
                self.create_acoustic_failure_overlay(analyze_all=analyze_all)
            else:
                self.chat_history.append("<span style='color:red;'>X Analysis returned no results</span>")
        except Exception as e:
            self.chat_history.append(f"<span style='color:red;'>X Acoustic failure analysis failed: {e}</span>")
            print(f"X Acoustic failure analysis failed: {e}")
            import traceback
            traceback.print_exc()

    def identify_failing_acoustic_spaces(self, analyze_all=False):
        """Identify spaces that are failing acoustic requirements, grouped by apartment type (show pass/fail)"""
        try:
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                return "No IFC file loaded. Please load an IFC file first."
            ifc_file = ifcopenshell.open(self.viewer.current_ifc_path)
            spaces = ifc_file.by_type("IfcSpace")
            if not spaces:
                return "No IfcSpace elements found in the IFC file."
            import re
            spaces_by_type = {}
            for space in spaces:
                name = getattr(space, 'Name', '') or ''
                apt_type = None
                match = re.search(r'([123])[_ ]?B(ED)?', name.upper())
                if match:
                    apt_type = f"{match.group(1)}B"
                if not apt_type:
                    apt_type = 'UNKNOWN'
                if apt_type not in spaces_by_type:
                    spaces_by_type[apt_type] = []
                spaces_by_type[apt_type].append(space)
            analysis_results = []
            total_analyzed = 0
            total_failures = 0
            total_passes = 0
            analysis_results.append("SEARCH Acoustic Analysis (by Apartment Type, Pass/Fail):")
            analysis_results.append("=" * 40)
            print(f"[DEBUG] Starting acoustic analysis for {len(spaces)} spaces, grouped by {len(spaces_by_type)} apartment types.")
            for apt_type, group_spaces in spaces_by_type.items():
                print(f"[DEBUG] Analyzing apartment type: {apt_type} ({len(group_spaces)} spaces)")
                if analyze_all:
                    spaces_to_analyze = group_spaces
                else:
                    spaces_to_analyze = self.select_spaces_for_analysis(group_spaces)
                analyzed_count = 0
                failing_spaces = []
                passing_spaces = []
                for i, space in enumerate(spaces_to_analyze):
                    try:
                        space_info = self.analyze_single_space(space, ifc_file)
                        if space_info:
                            analyzed_count += 1
                            failures = self.check_acoustic_failures(space_info)
                            severity = self.calculate_acoustic_severity(failures)
                            if severity in ["high", "critical"]:
                                failing_spaces.append((space_info, failures, severity))
                                print(f"[DEBUG] {space_info['name']} (ID: {space_info['global_id']}) - FAIL ({severity})")
                            else:
                                passing_spaces.append((space_info, failures, severity))
                                print(f"[DEBUG] {space_info['name']} (ID: {space_info['global_id']}) - PASS ({severity})")
                    except Exception as e:
                        print(f"Error analyzing space {i}: {e}")
                        continue
                total_analyzed += analyzed_count
                total_failures += len(failing_spaces)
                total_passes += len(passing_spaces)
                print(f"[DEBUG] Finished {apt_type}: {analyzed_count} analyzed, {len(passing_spaces)} pass, {len(failing_spaces)} fail.")
                analysis_results.append(f"\nBUILDING Apartment Type: {apt_type}")
                analysis_results.append(f"• Spaces analyzed: {analyzed_count}")
                analysis_results.append(f"• Passing spaces: {len(passing_spaces)}")
                analysis_results.append(f"• Failing spaces: {len(failing_spaces)}")
                if analyzed_count > 0:
                    analysis_results.append(f"• Failure rate: {(len(failing_spaces)/analyzed_count*100):.1f}%")
                for idx, (space_info, failures, severity) in enumerate(failing_spaces):
                    analysis_results.append(f"   X {space_info['name']} (ID: {space_info['global_id']}) [Severity: {severity}]")
                    for failure in failures:
                        analysis_results.append(f"      - {failure}")
                for idx, (space_info, failures, severity) in enumerate(passing_spaces):
                    analysis_results.append(f"   OK {space_info['name']} (ID: {space_info['global_id']}) [Severity: {severity}]")
            analysis_results.append("\nTREND_UP Combined Summary:")
            analysis_results.append(f"• Total spaces in file: {len(spaces)}")
            analysis_results.append(f"• Total spaces analyzed: {total_analyzed}")
            analysis_results.append(f"• Total passing spaces: {total_passes}")
            analysis_results.append(f"• Total failing spaces: {total_failures}")
            if total_analyzed > 0:
                analysis_results.append(f"• Overall failure rate: {(total_failures/total_analyzed*100):.1f}%")
                analysis_results.append(f"• Overall pass rate: {(total_passes/total_analyzed*100):.1f}%")
            print(f"[DEBUG] Acoustic analysis complete: {total_analyzed} analyzed, {total_passes} pass, {total_failures} fail.")
            return "\n".join(analysis_results)
        except Exception as e:
            return f"Error in acoustic failure analysis: {e}"

    def select_spaces_for_analysis(self, group_spaces, sample_ratio=0.4):
        """
        Select a sample of spaces for acoustic analysis, prioritizing key space types.
        Returns a list of selected spaces (max 40% of total, or min 1).
        """
        import random
        try:
            # Priority selection logic
            priority_spaces = []
            regular_spaces = []
            for space in group_spaces:
                space_name = getattr(space, 'Name', '') or ''
                name_upper = space_name.upper()
                priority_indicators = [
                    'LVL4', 'LVL5', 'LVL6',  # Higher floors
                    'ROOF', 'TOP', 'PENTHOUSE',
                    'MECH', 'HVAC', 'EQUIPMENT',
                    'CORRIDOR', 'HALLWAY',
                    'LOBBY', 'ENTRANCE',
                    'KITCHEN', 'BATHROOM',
                    'ELEVATOR', 'STAIR'
                ]
                is_priority = any(indicator in name_upper for indicator in priority_indicators)
                if is_priority:
                    priority_spaces.append(space)
                else:
                    regular_spaces.append(space)
            # 40% of group_spaces or at least 1
            n_total = max(1, int(len(group_spaces) * sample_ratio))
            # Get up to 15 priority, rest as random from regulars
            max_priority = min(len(priority_spaces), n_total)
            max_regular = n_total - max_priority
            selected_spaces = priority_spaces[:max_priority]
            if regular_spaces and max_regular > 0:
                selected_spaces.extend(random.sample(regular_spaces, min(max_regular, len(regular_spaces))))
            # If not enough, fill from remaining
            if len(selected_spaces) < n_total:
                remaining = [s for s in group_spaces if s not in selected_spaces]
                selected_spaces.extend(remaining[:n_total - len(selected_spaces)])
            print(f"SEARCH Smart selection: {len(priority_spaces)} priority, {len(regular_spaces)} regular, {len(selected_spaces)} selected.")
            return selected_spaces
        except Exception as e:
            print(f"Error in smart space selection: {e}")
            # fallback: just the first N
            return group_spaces[:max(1, int(len(group_spaces) * sample_ratio))]


    def create_acoustic_failure_overlay(self, analyze_all=False):
        """Create a visual overlay highlighting ALL spaces (pass/fail)"""
        try:
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                self.chat_history.append("<span style='color:red;'>X No IFC file loaded for overlay</span>")
                return
            self.chat_history.append("<span style='color:#4CAF50;'><b>ART Creating Acoustic Overlay...</b></span>")
            all_spaces = self.get_failing_spaces_data(analyze_all=analyze_all)
            if not all_spaces:
                self.chat_history.append("<span style='color:#4CAF50;'>OK No spaces found</span>")
                return
            self.apply_acoustic_failure_colors(all_spaces)
            # Count passing and failing
            passing = sum(1 for s in all_spaces if s['severity'] == 'low')
            failing = sum(1 for s in all_spaces if s['severity'] in ['high', 'critical'])
            self.show_acoustic_failure_legend(all_spaces)
            self.chat_history.append(f"<span style='color:#4CAF50;'>OK Acoustic overlay applied! {len(all_spaces)} spaces highlighted: {passing} passing, {failing} failing</span>")
            self.chat_history.append("<span style='color:#FF9800;'>IDEA Hover over colored spaces in the 3D viewer to see detailed information</span>")
        except Exception as e:
            self.chat_history.append(f"<span style='color:red;'>X Overlay creation failed: {e}</span>")
            print(f"X Overlay error: {e}")
            import traceback
            traceback.print_exc()

    def show_acoustic_failure_legend(self, all_spaces):
        """Show legend for acoustic overlay (pass/fail)"""
        try:
            # Count by severity
            severity_counts = {}
            for fs in all_spaces:
                severity = fs['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            legend_html = """
            <div style="background: #2C2C2C; padding: 15px; border-radius: 10px; color: white; margin: 10px 0;">
                <h3 style="margin-top: 0; color: #FF9500;">MUSIC Acoustic Overlay Legend</h3>
                <div style="display: flex; flex-direction: column; gap: 8px;">
            """
            severity_info = {
                "critical": {"color": "rgb(255, 0, 0)", "desc": "Critical (Fail) - Multiple severe issues"},
                "high": {"color": "rgb(255, 128, 0)", "desc": "High (Fail) - Significant acoustic problems"},
                "medium": {"color": "rgb(0, 255, 0)", "desc": "Medium (Pass) - Moderate issues"},
                "low": {"color": "rgb(0, 255, 0)", "desc": "Low (Pass) - No or minor issues"}
            }
            for severity, info in severity_info.items():
                count = severity_counts.get(severity, 0)
                if count > 0:
                    legend_html += f"""
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div style="width: 20px; height: 20px; background: {info['color']}; border-radius: 3px;"></div>
                        <span>{info['desc']}: {count} spaces</span>
                    </div>
                    """
            legend_html += """
                </div>
                <div style="margin-top: 10px; font-size: 13px; color: #CCC;">
                    IDEA Green = Pass, Yellow/Orange/Red = Fail
                </div>
            </div>
            """
            self.chat_history.append(f"<span style='color:#4CAF50;'><b>CHART Acoustic Overlay Summary:</b></span>")
            for severity, count in severity_counts.items():
                label = "PASS" if severity == "low" else "FAIL"
                self.chat_history.append(f"<span style='color:#FF9500;'>• {severity.title()} ({label}): {count} spaces</span>")
        except Exception as e:
            print(f"X Error showing legend: {e}")

    def get_failing_spaces_data(self, analyze_all=False):
        """Get data for all failing acoustic spaces"""
        try:
            ifc_file = ifcopenshell.open(self.viewer.current_ifc_path)
            spaces = ifc_file.by_type("IfcSpace")
            if not spaces:
                print("No IfcSpace elements found")
                return []
            failing_spaces = []
            # Use smart space selection for consistency
            if analyze_all:
                spaces_to_analyze = spaces
            else:
                spaces_to_analyze = self.select_spaces_for_analysis(spaces)
            print(f"[DEBUG] Analyzing {len(spaces_to_analyze)} spaces (analyze_all={analyze_all})")
            for i, space in enumerate(spaces_to_analyze):
                try:
                    space_info = self.analyze_single_space(space, ifc_file)
                    if space_info:
                        failures = self.check_acoustic_failures(space_info)
                        if failures:
                            # Determine severity level and color
                            severity = self.calculate_acoustic_severity(failures)
                            color = self.get_severity_color(severity)
                            failing_spaces.append({
                                'space': space,
                                'space_info': space_info,
                                'failures': failures,
                                'severity': severity,
                                'color': color,
                                'global_id': space_info['global_id']
                            })
                except Exception as e:
                    print(f"Error analyzing space {i} for failing data: {e}")
                    continue
            return failing_spaces
        except Exception as e:
            print(f"Error getting failing spaces data: {e}")
            return []

    def apply_acoustic_failure_colors(self, all_spaces):
        """Apply color coding to ALL spaces in the viewer according to pass/fail result (clean, not overlay)."""
        try:
            colored_count = 0
            # Create mapping of space Global IDs to result data
            space_result_map = {fs['global_id']: fs for fs in all_spaces}
            # Apply colors to ALL spaces in the viewer
            for ais_shape, ifc_elem in self.viewer.shape_to_ifc.items():
                if ifc_elem.is_a() == "IfcSpace":
                    global_id = getattr(ifc_elem, "GlobalId", "")
                    if global_id in space_result_map:
                        result_data = space_result_map[global_id]
                        color = result_data['color']
                        # Apply color to the shape
                        color_obj = Quantity_Color(color[0], color[1], color[2], Quantity_TOC_RGB)
                        self.viewer._display.Context.SetColor(ais_shape, color_obj, False)
                        # Make it fully opaque for clean result
                        self.viewer._display.Context.SetTransparency(ais_shape, 0.0, False)
                        colored_count += 1
                    else:
                        # If not analyzed, optionally set to a neutral color (e.g., gray)
                        color_obj = Quantity_Color(0.7, 0.7, 0.7, Quantity_TOC_RGB)
                        self.viewer._display.Context.SetColor(ais_shape, color_obj, False)
                        self.viewer._display.Context.SetTransparency(ais_shape, 0.7, False)
            # Update the display
            self.viewer._display.Context.UpdateCurrentViewer()
            self.viewer._display.Repaint()
            print(f"OK Applied acoustic result colors to {colored_count} spaces (clean, not overlay)")
        except Exception as e:
            print(f"X Error applying acoustic result colors: {e}")

    def show_graph_viewer(self):
        print("[DEBUG] show_graph_viewer() - Starting...")
        try:
            if hasattr(self, "graph_viewer"):
                print("[DEBUG] show_graph_viewer() - GraphViewer already exists, showing...")
                self.graph_viewer.show()
                self.graph_viewer.raise_()
                print("[DEBUG] show_graph_viewer() - Existing GraphViewer shown and raised")
            else:
                print("[DEBUG] show_graph_viewer() - Creating new GraphViewer...")
                self.graph_viewer = GraphViewer(self)
                print("[DEBUG] show_graph_viewer() - GraphViewer created successfully")
                
                print("[DEBUG] show_graph_viewer() - Adding GraphViewer to dock widget...")
                self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.graph_viewer)
                print("[DEBUG] show_graph_viewer() - GraphViewer added to dock widget")
                
                print("[DEBUG] show_graph_viewer() - Showing GraphViewer...")
                self.graph_viewer.show()
                print("[DEBUG] show_graph_viewer() - GraphViewer show() called")
                
                # Check if graph loaded successfully
                print("[DEBUG] show_graph_viewer() - Checking available graphs...")
                available_graphs = self.graph_viewer.get_available_graphs()
                if available_graphs:
                    print(f"OK Available graph files: {available_graphs}")
                else:
                    print("WARNING No graph files found in models/Graphs/ directory")
                
                print("[DEBUG] show_graph_viewer() - Method completed successfully")
        except Exception as e:
            print(f"[DEBUG] show_graph_viewer() - Error: {e}")
            import traceback
            traceback.print_exc()


    def upload_to_neo4j(self):
        """Upload IFC data to Neo4j database"""
        try:
            # Check if IFC file is loaded
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                QMessageBox.warning(self, "Warning", "No IFC file loaded. Please load an IFC file first.")
                return
            
            # Extract model data
            data = self.viewer.extract_model_data()
            if not data:
                QMessageBox.warning(self, "Upload Failed", "No IFC data to upload.")
                return
            
            # Check if Neo4j connector exists
            if not hasattr(self, 'neo4j') or not self.neo4j:
                QMessageBox.warning(self, "Neo4j Error", "Neo4j connector not available.")
                return
            
            try:
                # Test connection first (this will trigger lazy connection)
                if not self.neo4j.is_connected():
                    print("[DEBUG] Attempting to connect to Neo4j...")
                    self.neo4j._connect()
                
                print(f"[DEBUG] Attempting to upload {len(data)} elements to Neo4j...")
                
                # Insert the data
                self.neo4j.insert_all(data)
                
                print(f"[DEBUG] Successfully uploaded {len(data)} elements to Neo4j")
                QMessageBox.information(self, "Success", f"IFC data uploaded to Neo4j successfully.\nUploaded {len(data)} elements.")
                
                # Create and show Neo4j viewer
                self.neo4j_viewer = Neo4jViewer(self)
                self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.neo4j_viewer)
                self.neo4j_viewer.show()
                
                # Show graph viewer after successful upload
                print("[DEBUG] upload_to_neo4j() - Calling show_graph_viewer after successful upload...")
                self.show_graph_viewer()
                print("[DEBUG] upload_to_neo4j() - show_graph_viewer call completed")
                
            except Exception as e:
                error_msg = f"Failed to upload data to Neo4j: {str(e)}"
                print(f"[DEBUG] Neo4j upload error: {error_msg}")
                
                # Provide more specific error messages
                if "Connection refused" in str(e) or "Failed to connect" in str(e):
                    QMessageBox.warning(self, "Neo4j Connection Error", 
                        "Could not connect to Neo4j database.\n\n"
                        "Please ensure:\n"
                        "1. Neo4j database is running\n"
                        "2. Neo4j is accessible at bolt://localhost:7687\n"
                        "3. Username: neo4j, Password: 123456789\n"
                        "4. Firewall allows connection to port 7687")
                else:
                    QMessageBox.warning(self, "Neo4j Upload Error", error_msg)
                
        except Exception as e:
            error_msg = f"Upload failed: {str(e)}"
            print(f"[DEBUG] General upload error: {error_msg}")
            QMessageBox.warning(self, "Error", error_msg)

    def export_ifc_to_excel(self):
        """Export IFC data to Excel"""
        try:
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                QMessageBox.warning(self, "Warning", "No IFC file loaded. Please load an IFC file first.")
                return
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export IFC Data", "", "Excel Files (*.xlsx);;All Files (*)"
            )
            
            if file_path:
                # Extract data from IFC file
                ifc_file = ifcopenshell.open(self.viewer.current_ifc_path)
                
                # Get all spaces
                spaces = ifc_file.by_type('IfcSpace')
                
                # Create DataFrame
                data = []
                for space in spaces:
                    space_data = {
                        'GlobalId': getattr(space, 'GlobalId', 'N/A'),
                        'Name': getattr(space, 'Name', 'N/A'),
                        'LongName': getattr(space, 'LongName', 'N/A'),
                        'ObjectType': getattr(space, 'ObjectType', 'N/A'),
                        'Description': getattr(space, 'Description', 'N/A')
                    }
                    data.append(space_data)
                
                df = pd.DataFrame(data)
                df.to_excel(file_path, index=False)
                
                QMessageBox.information(self, "Success", f"IFC data exported to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Export failed:\n{e}")

    def capture_screen(self):
        """Capture the current viewport as a screenshot"""
        try:
            from scripts.core.capture import capture_viewport
            capture_viewport(self.viewer, self)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Screen capture failed:\n{e}")

    def show_inputs_panel(self):
        """Show the inputs panel (left panel)"""
        self.left_panel.setVisible(True)
        self.left_panel.raise_()  # Bring to front
        self.chat_history.append("<span style='color:#4CAF50;'>OK Input panel restored</span>")

    def get_geometry_summary(self):
        # For example, count elements by type:
        from collections import Counter
        summary = Counter([ifc_elem.is_a() for ifc_elem in self.viewer.shape_to_ifc.values()])
        # Or get a list of all wall names:
        wall_names = [e.Name for e in self.viewer.shape_to_ifc.values() if e.is_a() == "IfcWall"]
        # Return as a string, JSON, or dict, whatever the LLM can use
        return {
            "element_counts": dict(summary),
            "wall_names": wall_names
    }
   
   
    def on_chatbot_send(self):
        question = self.chat_input.text().strip()
        if not question:
            return
        self.chat_history.append(f"<b>You:</b> {question}")
        self.chat_input.clear()
        try:
            # --- Extract live IFC model data for context (JSON, InfoDock, hardcoded) ---
            model_context = ""
            try:
                if hasattr(self, 'ifc_json_data') and self.ifc_json_data:
                    # Debug: Show what's in ifc_json_data
                    print(f"[DEBUG] ifc_json_data type: {type(self.ifc_json_data)}")
                    print(f"[DEBUG] ifc_json_data: {self.ifc_json_data}")
                    
                    # Handle different data types
                    if isinstance(self.ifc_json_data, list):
                        print(f"[DEBUG] ifc_json_data length: {len(self.ifc_json_data)}")
                        if len(self.ifc_json_data) > 0:
                            print(f"[DEBUG] First element type: {type(self.ifc_json_data[0])}")
                            print(f"[DEBUG] First element: {self.ifc_json_data[0]}")
                    elif isinstance(self.ifc_json_data, dict):
                        print(f"[DEBUG] ifc_json_data is a dictionary with keys: {list(self.ifc_json_data.keys())}")
                        # Check if it's an error dictionary
                        if 'error' in self.ifc_json_data:
                            print(f"[DEBUG] JSON parsing failed: {self.ifc_json_data['error']}")
                            # Use fallback IFC analysis
                            model_context = self.get_fallback_ifc_context()
                        else:
                            # If it's a dict, try to find a list of elements
                            data_to_process = []
                            for key, value in self.ifc_json_data.items():
                                if isinstance(value, list):
                                    data_to_process.extend(value)
                                elif isinstance(value, dict):
                                    data_to_process.append(value)
                    elif isinstance(self.ifc_json_data, str):
                        print(f"[DEBUG] ifc_json_data is a string: {self.ifc_json_data[:100]}...")
                    else:
                        print(f"[DEBUG] ifc_json_data is unknown type: {type(self.ifc_json_data)}")
                    
                    # Only process if we have valid data (not an error)
                    if not model_context:  # If fallback wasn't used
                        import re
                        space_match = re.search(r'(LVL\d+_[123]B_\d+|[A-Za-z0-9\-]{22})', question)
                        if space_match:
                            space_id = space_match.group(1)
                            elem = self.get_json_element_by_name_or_id(space_id)
                            if elem:
                                model_context += f"Specific Element Data:\n{elem}\n"
                        else:
                            # Summarize all spaces/elements with apartment types
                            summary = []
                            apartment_counts = {}
                            
                            # Handle different data types for ifc_json_data
                            if isinstance(self.ifc_json_data, list):
                                data_to_process = self.ifc_json_data
                            elif isinstance(self.ifc_json_data, dict):
                                # If it's a dict, try to find a list of elements
                                data_to_process = []
                                for key, value in self.ifc_json_data.items():
                                    if isinstance(value, list):
                                        data_to_process.extend(value)
                                    elif isinstance(value, dict):
                                        data_to_process.append(value)
                            else:
                                # If it's not a list or dict, skip processing
                                data_to_process = []
                                print(f"[DEBUG] Skipping non-list/dict ifc_json_data: {type(self.ifc_json_data)}")
                            
                            for e in data_to_process:
                                # Add type checking to ensure e is a dictionary
                                if isinstance(e, dict) and e.get('type') == 'IfcSpace':
                                    apt_type = e.get('apartment_type', 'Unknown')
                                    apartment_counts[apt_type] = apartment_counts.get(apt_type, 0) + 1
                                    summary.append(f"{e.get('name', 'Unknown')} (ID: {e.get('global_id', 'Unknown')}), Type: {apt_type}, Vol: {e.get('hardcoded', {}).get('volume', 'N/A')} m³, Area: {e.get('hardcoded', {}).get('area', 'N/A')} m², H: {e.get('hardcoded', {}).get('height', 'N/A')} m, RT60: {e.get('properties', {}).get('RT60', 'N/A')}, SPL: {e.get('properties', {}).get('SPL', 'N/A')}")
                                elif isinstance(e, str):
                                    # Handle string elements (skip or log)
                                    print(f"[DEBUG] Skipping string element: {e[:50]}...")
                                else:
                                    # Handle other data types
                                    print(f"[DEBUG] Skipping non-dict element: {type(e)}")
                            
                            model_context += f"Apartment Type Distribution: {apartment_counts}\n"
                            model_context += f"Total Spaces: {len(summary)}\n"
                            model_context += "Space Details:\n" + "\n".join(summary)
                else:
                    # No ifc_json_data available, use fallback
                    print("[DEBUG] No ifc_json_data available, using fallback")
                    model_context = self.get_fallback_ifc_context()
            except Exception as e:
                print(f"[DEBUG] Error processing IFC data: {e}")
                model_context = self.get_fallback_ifc_context()
            
            # Add InfoDock data if available
            if hasattr(self, 'info_dock') and self.info_dock and hasattr(self.info_dock, 'current_element_data') and self.info_dock.current_element_data:
                model_context += f"\nCurrently Selected Element: {self.info_dock.current_element_data}\n"
            # Fallback if no IFC loaded
            if not model_context:
                model_context = "No IFC file loaded."
            
            # --- Enhanced prompt with detailed recommendations ---
            prompt = f"""
You are analyzing a specific IFC building model that is currently loaded. Use the actual model data below to answer the user's question.

CURRENT IFC MODEL DATA:
{model_context}

USER QUESTION: {question}

INSTRUCTIONS:
1. ALWAYS start your answer by referencing the actual loaded IFC model data above
2. Mention specific apartment types, space counts, and properties from the model
3. If the question asks about materials, reference the actual materials in your loaded model
4. If the question asks about acoustic properties, reference the actual RT60, SPL values from your model
5. Give specific answers about the actual loaded geometry, not generic responses
6. If the question mentions specific spaces, use that exact data
7. If asking about apartment types, reference the actual counts and properties from the model

FOR ACOUSTIC ANALYSIS AND RECOMMENDATIONS:
- If acoustic failures are detected, provide DETAILED recommendations including:
  * Specific material upgrades with absorption coefficients
  * Construction improvements (wall thickness, insulation, etc.)
  * Room geometry modifications
  * HVAC system adjustments
  * Furniture and acoustic treatment suggestions
  * Cost estimates for improvements
  * Priority ranking of recommendations
  * Expected acoustic performance improvements

- Always explain WHY each recommendation will help
- Provide specific absorption coefficients and material properties
- Include both immediate fixes and long-term solutions
- Consider the specific apartment type and use case

FOR ACOUSTIC IMPROVEMENT QUESTIONS:
- If the user asks about "suggestions", "improvements", "solutions", or "recommendations":
  * Provide a comprehensive list of specific acoustic treatments
  * Include material specifications with absorption coefficients
  * Give cost estimates and installation timelines
  * Explain the expected performance improvements
  * Prioritize recommendations by urgency and impact
  * Include both DIY and professional installation options
  * Provide supplier recommendations where appropriate

EXAMPLE: If asked about materials, say "In your loaded IFC model, I can see [specific materials/properties from the model data]..." instead of generic material advice.

Please provide a detailed, model-specific answer that references the actual loaded geometry.
"""
            
            # Debug print the prompt
            print(f"\n[DEBUG] Simplified Prompt (IFC-only):")
            print(f"[DEBUG] {prompt}")
            
            from server.config import client, completion_model
            response = client.chat.completions.create(
                model=completion_model,
                messages=[
                    {"role": "system", "content": "You are an architectural acoustics advisor analyzing a SPECIFIC IFC building model that is currently loaded. You MUST ALWAYS reference the actual loaded geometry, spaces, apartment types, and properties from the model data provided. NEVER give generic acoustic advice - always start with what you can see in the loaded model. If the model data shows specific materials, properties, or spaces, reference them directly. When providing acoustic recommendations, ALWAYS include: 1) Specific material upgrades with absorption coefficients, 2) Construction improvements, 3) Room geometry modifications, 4) HVAC adjustments, 5) Acoustic treatment suggestions, 6) Cost estimates, 7) Priority ranking, and 8) Expected performance improvements. Explain WHY each recommendation will help and provide both immediate fixes and long-term solutions."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content.strip()
            
            # Debug print the final AI response
            print(f"\n[DEBUG] Final AI Response:")
            print(f"[DEBUG] {answer}")
            
            self.chat_history.append(f"<span style='color:#E98801;'><b>AI:</b> {answer}</span>")
        except Exception as e:
            self.chat_history.append(f"<span style='color:red;'>[Error]: {e}</span>")
            print(f"X Chatbot error: {e}")
            import traceback
            traceback.print_exc()

    def get_comprehensive_ifc_analysis(self):
        """Get comprehensive analysis of the IFC file including all element types"""
        try:
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                return "No IFC file loaded. Please load an IFC file first."
            
            # Analyze the full IFC file (not just loaded elements)
            element_counts = self.viewer.analyze_ifc_file(self.viewer.current_ifc_path)
            
            if not element_counts:
                return "Could not analyze IFC file."
            
            # Create detailed analysis
            analysis = []
            analysis.append("CHART Complete IFC Element Analysis:")
            analysis.append("=" * 40)
            
            # Sort by count (highest first)
            sorted_elements = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)
            
            for elem_type, count in sorted_elements:
                analysis.append(f"• {elem_type}: {count}")
            
            # Enhanced space analysis with acoustic properties
            space_count = element_counts.get('IfcSpace', 0)
            if space_count > 0:
                analysis.append(f"\nBUILDING Enhanced Spaces Analysis:")
                analysis.append(f"• Total IfcSpace elements: {space_count}")
                
                # Get detailed space analysis with acoustic properties
                space_analysis = self.get_detailed_space_analysis()
                analysis.extend(space_analysis)
            
            # Building structure analysis
            wall_count = element_counts.get('IfcWall', 0)
            slab_count = element_counts.get('IfcSlab', 0)
            column_count = element_counts.get('IfcColumn', 0)
            beam_count = element_counts.get('IfcBeam', 0)
            
            if wall_count or slab_count or column_count or beam_count:
                analysis.append(f"\n🏗️ Building Structure:")
                if wall_count:
                    analysis.append(f"• Walls: {wall_count}")
                if slab_count:
                    analysis.append(f"• Slabs/Floors: {slab_count}")
                if column_count:
                    analysis.append(f"• Columns: {column_count}")
                if beam_count:
                    analysis.append(f"• Beams: {beam_count}")
            
            # Openings
            door_count = element_counts.get('IfcDoor', 0)
            window_count = element_counts.get('IfcWindow', 0)
            
            if door_count or window_count:
                analysis.append(f"\n🚪 Openings:")
                if door_count:
                    analysis.append(f"• Doors: {door_count}")
                if window_count:
                    analysis.append(f"• Windows: {window_count}")
            
            return "\n".join(analysis)
            
        except Exception as e:
            return f"Error analyzing IFC file: {e}"

    def get_detailed_space_analysis(self):
        """Get detailed analysis of spaces with acoustic properties and relationships"""
        try:
            ifc_file = ifcopenshell.open(self.viewer.current_ifc_path)
            spaces = ifc_file.by_type("IfcSpace")
            
            analysis = []
            
            # Analyze each space in detail
            for i, space in enumerate(spaces[:10]):  # Limit to first 10 for performance
                space_info = self.analyze_single_space(space, ifc_file)
                if space_info:
                    analysis.append(f"\n📍 Space {i+1}: {space_info['name']}")
                    analysis.append(f"   • Global ID: {space_info['global_id']}")
                    analysis.append(f"   • Volume: {space_info['volume']} m³")
                    analysis.append(f"   • Area: {space_info['area']} m²")
                    analysis.append(f"   • Height: {space_info['height']} m")
                    
                    # Acoustic properties
                    if space_info['acoustic_properties']:
                        analysis.append(f"   • Acoustic Properties:")
                        for prop, value in space_info['acoustic_properties'].items():
                            analysis.append(f"     - {prop}: {value}")
                    
                    # Surrounding elements
                    if space_info['surrounding_elements']:
                        analysis.append(f"   • Surrounding Elements:")
                        for elem_type, count in space_info['surrounding_elements'].items():
                            analysis.append(f"     - {elem_type}: {count}")
                    
                    # Acoustic risk assessment
                    if space_info['acoustic_risk']:
                        analysis.append(f"   • Acoustic Risk: {space_info['acoustic_risk']}")
            
            # Summary statistics
            total_volume = sum(space_info.get('volume', 0) for space_info in [self.analyze_single_space(space, ifc_file) for space in spaces[:10]])
            avg_volume = total_volume / min(len(spaces), 10)
            
            analysis.append(f"\nTREND_UP Summary Statistics (first 10 spaces):")
            analysis.append(f"• Average volume: {avg_volume:.1f} m³")
            analysis.append(f"• Total spaces analyzed: {min(len(spaces), 10)}")
            
            return analysis
            
        except Exception as e:
            return [f"Error in detailed space analysis: {e}"]

    def analyze_single_space(self, space, ifc_file):
        """Analyze a single space with its acoustic properties and relationships"""
        try:
            # Handle None names properly
            space_name = getattr(space, 'Name', None)
            if space_name is None:
                space_name = 'Unnamed'
            space_info = {
                'name': space_name,
                'global_id': getattr(space, 'GlobalId', 'Unknown'),
                'volume': 0,
                'area': 0,
                'height': 0,
                'acoustic_properties': {},
                'surrounding_elements': {},
                'acoustic_risk': 'Unknown'
            }
            # Hardcoded apartment dimensions based on type
            apartment_dimensions = {
                '1B': {'volume': 58, 'area': 19.33, 'height': 3.0},
                '2B': {'volume': 81, 'area': 27.00, 'height': 3.0},
                '3B': {'volume': 108, 'area': 36.00, 'height': 3.0},
            }
            # Robust apartment type detection
            apartment_type = self.detect_apartment_type(space_name)
            if not apartment_type:
                apartment_type = '2B'  # Default
            # Apply hardcoded dimensions
            if apartment_type in apartment_dimensions:
                dims = apartment_dimensions[apartment_type]
                space_info['volume'] = dims['volume']
                space_info['area'] = dims['area']
                space_info['height'] = dims['height']
                space_info['apartment_type'] = apartment_type
            # Extract geometric properties (as fallback if hardcoded not available)
            if hasattr(space, 'Representation') and space.Representation:
                try:
                    shape = ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(), space)
                    if shape and hasattr(shape, 'geometry'):
                        bbox = shape.geometry.BoundingBox()
                        if bbox:
                            width = bbox.Xmax() - bbox.Xmin()
                            height = bbox.Ymax() - bbox.Ymin()
                            depth = bbox.Zmax() - bbox.Zmin()
                            if space_info['volume'] == 0:
                                space_info['volume'] = width * height * depth
                                space_info['area'] = width * depth
                                space_info['height'] = height
                except:
                    pass
            # Extract acoustic properties from property sets
            if hasattr(space, 'IsDefinedBy'):
                for rel in space.IsDefinedBy:
                    if rel.is_a('IfcRelDefinesByProperties'):
                        props = rel.RelatingPropertyDefinition
                        if props.is_a('IfcPropertySet'):
                            for prop in props.HasProperties:
                                if prop.is_a('IfcPropertySingleValue'):
                                    prop_name = prop.Name.lower() if prop.Name else ""
                                    prop_value = getattr(prop.NominalValue, 'wrappedValue', None)
                                    if any(acoustic_term in prop_name for acoustic_term in ['rt60', 'spl', 'acoustic', 'sound', 'noise', 'reverberation', 'laeq', 'l(a)eq']):
                                        space_info['acoustic_properties'][prop.Name or 'Unknown'] = prop_value
            # Find surrounding elements (walls, slabs, etc.)
            surrounding_elements = self.find_surrounding_elements(space, ifc_file)
            space_info['surrounding_elements'] = surrounding_elements
            # Assess acoustic risk based on available data
            space_info['acoustic_risk'] = self.assess_acoustic_risk(space_info)
            return space_info
        except Exception as e:
            print(f"Error analyzing space: {e}")
            return None

    def find_surrounding_elements(self, space, ifc_file):
        """Find elements that surround or are related to a space"""
        try:
            surrounding = {}
            
            # Method 1: Check IfcRelSpaceBoundary
            if hasattr(space, 'BoundedBy'):
                for rel in space.BoundedBy:
                    if hasattr(rel, 'RelatedBuildingElement'):
                        elem = rel.RelatedBuildingElement
                        elem_type = elem.is_a()
                        surrounding[elem_type] = surrounding.get(elem_type, 0) + 1
            
            # Method 2: Check IfcRelContainedInSpatialStructure
            if hasattr(space, 'ContainsElements'):
                for rel in space.ContainsElements:
                    for elem in rel.RelatedElements:
                        elem_type = elem.is_a()
                        surrounding[elem_type] = surrounding.get(elem_type, 0) + 1
            
            # Method 3: Check by spatial hierarchy
            all_products = ifc_file.by_type('IfcProduct')
            for product in all_products:
                if hasattr(product, 'ContainedInStructure'):
                    for rel in product.ContainedInStructure:
                        if rel.RelatingStructure.GlobalId == space.GlobalId:
                            elem_type = product.is_a()
                            surrounding[elem_type] = surrounding.get(elem_type, 0) + 1
            
            return surrounding
            
        except Exception as e:
            print(f"Error finding surrounding elements: {e}")
            return {}

    def assess_acoustic_risk(self, space_info):
        """Assess acoustic risk based on space properties"""
        try:
            risk_factors = []
            
            # Volume-based risk
            volume = space_info.get('volume', 0)
            if volume > 1000:  # Large spaces
                risk_factors.append("Large volume may cause acoustic issues")
            elif volume < 50:  # Very small spaces
                risk_factors.append("Small volume may cause sound pressure issues")
            
            # Surrounding elements risk
            surrounding = space_info.get('surrounding_elements', {})
            wall_count = surrounding.get('IfcWall', 0)
            door_count = surrounding.get('IfcDoor', 0)
            window_count = surrounding.get('IfcWindow', 0)
            
            if door_count > 3:
                risk_factors.append("High number of doors may cause sound leakage")
            if window_count > 2:
                risk_factors.append("High number of windows may cause sound transmission")
            if wall_count < 4:
                risk_factors.append("Insufficient walls for proper acoustic isolation")
            
            # Acoustic properties risk
            acoustic_props = space_info.get('acoustic_properties', {})
            if not acoustic_props:
                risk_factors.append("No acoustic properties defined")
            else:
                # Check for specific acoustic issues
                for prop_name, value in acoustic_props.items():
                    if 'rt60' in prop_name.lower() and value:
                        try:
                            rt60_val = float(value)
                            if rt60_val > 2.0:
                                risk_factors.append(f"High RT60 ({rt60_val}s) - poor acoustic performance")
                            elif rt60_val < 0.3:
                                risk_factors.append(f"Very low RT60 ({rt60_val}s) - may be too dead")
                        except:
                            pass
            
            if not risk_factors:
                return "Low risk - appears to have good acoustic properties"
            else:
                return "; ".join(risk_factors[:3])  # Limit to 3 most important
            
        except Exception as e:
            return f"Risk assessment error: {e}"

    def ask_openai_direct(question: str) -> str:
        try:
            response = client.chat.completions.create(
                model=completion_model,
                messages=[
                    {"role": "system", "content": "You are an assistant for acoustic comfort and architectural acoustics."},
                    {"role": "user", "content": question}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI error]: {e}"


    def handle_ifc_upload(self):
        """Handle IFC file upload with enhanced rendering"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select IFC File", "", "IFC Files (*.ifc);;All Files (*)"
        )
        if file_path:
            try:
                # First analyze the IFC file using the viewer's method
                element_counts = self.viewer.analyze_ifc_file(file_path)
                
                # Load parsed data for enhanced InfoDock
                self.info_dock.load_parsed_data(file_path)
                
                # Load IFC file
                success = self.viewer.load_ifc_file(file_path)
                if success:
                    # Enable refresh button
                    self.refresh_btn.setEnabled(True)
                    # Enable reset button
                    self.reset_btn.setEnabled(True)
                    # Show success message
                    self.chat_history.append(f"OK IFC file loaded: {os.path.basename(file_path)}")
                    
                    # Show model statistics
                    summary = self.viewer.get_ifc_summary()
                    if summary:
                        stats_text = f"CHART Model Statistics:\n"
                        element_counts = {}
                        for item in summary:
                            elem_type = item['type']
                            element_counts[elem_type] = element_counts.get(elem_type, 0) + 1
                        
                        for elem_type, count in element_counts.items():
                            stats_text += f"• {elem_type}: {count}\n"
                        
                        self.chat_history.append(stats_text)
                else:
                    QMessageBox.warning(self, "Error", "Failed to load IFC file")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading IFC file: {str(e)}")

    def on_evaluate_clicked(self):
        """Handle evaluate button click with enhanced ML integration"""
        # Hide the left panel (input panel) when evaluation starts
        self.left_panel.setVisible(False)
        
        # 1. Get user inputs
        user_input = {
            "apartment_type": self.geo_apartment.currentText(),
            "zone": self.zone_box.currentText(),  # Use actual zone from dropdown
            "wall_material": self.geo_wall.currentText(),
            "window_material": self.geo_window.currentText(),
            "floor_material": self.geo_floor.currentText(),  # Add floor material
            "floor_level": 2,  # Replace with an actual floor input if available
            "activity": self.activity_box.currentText(),
            "time_period": self.time_box.currentText(),  # Add time period
        }

        # 2. Get geometry data from InfoDock if available
        geometry_data = None
        if hasattr(self, 'info_dock') and self.info_dock and hasattr(self.info_dock, 'current_element_data'):
            geometry_data = self.info_dock.current_element_data
            if geometry_data:
                print("🎯 Using geometry data for enhanced ML prediction")

        # 3. Build a natural language prompt
        user_question = (
            f"Evaluate acoustic comfort for a {user_input['apartment_type']} apartment "
            f"in zone {user_input['zone']}, with {user_input['wall_material']} walls, "
            f"{user_input['window_material']} windows, and {user_input['floor_material']} floor. "
            f"Activity is {user_input['activity']} during {user_input['time_period']} period."
        )

        # 4. Show user input in chat panel
        self.chat_history.append(f"<b>You:</b> {user_question}")

        # 5. Call enhanced ML pipeline with geometry data
        try:
            from scripts.core.acoustic_pipeline import run_pipeline

            result = run_pipeline(user_input, user_question, geometry_data)
            print(f"🎯 Pipeline result: {result}")
            
            # Handle the result structure properly
            if isinstance(result, dict):
                # Check if it's an enhanced result
                if result.get("enhanced") and isinstance(result.get("result"), dict):
                    enhanced_result = result["result"]
                    
                    # Display SQL results first
                    if "sql_result" in enhanced_result:
                        sql_result = enhanced_result["sql_result"]
                        if isinstance(sql_result, dict) and "summary" in sql_result:
                            self.chat_history.append(f"<span style='color:#2196F3;'><b>CHART SQL Analysis:</b> {sql_result['summary']}</span>")
                        elif isinstance(sql_result, str):
                            self.chat_history.append(f"<span style='color:#2196F3;'><b>CHART SQL Analysis:</b> {sql_result}</span>")
                    
                    # Display comfort prediction
                    comfort_pred = enhanced_result.get("comfort_prediction")
                    if comfort_pred and "error" not in comfort_pred:
                        comfort_msg = f"🎯 <b>ML Comfort Prediction:</b> {comfort_pred['comfort_score']} (Confidence: {comfort_pred['confidence']:.1%})"
                        self.chat_history.append(f"<span style='color:#4CAF50;'>{comfort_msg}</span>")
                    elif comfort_pred and "error" in comfort_pred:
                        self.chat_history.append(f"<span style='color:#FF9800;'>WARNING ML Prediction: {comfort_pred['error']}</span>")
                    
                    # Display recommendations
                    recommendations = enhanced_result.get("recommendations", [])
                    if recommendations:
                        rec_msg = "<b>IDEA ML Recommendations:</b><br>"
                        for rec in recommendations:
                            rec_msg += f"• {rec}<br>"
                        self.chat_history.append(f"<span style='color:#FF9800;'>{rec_msg}</span>")
                    
                    # Display geometry analysis summary
                    geometry_analysis = enhanced_result.get("geometry_analysis", {})
                    if geometry_analysis:
                        geo_msg = f"<b>🏗️ Geometry Data:</b> Type: {geometry_analysis.get('element_type', 'Unknown')}"
                        if geometry_analysis.get('rt60'):
                            geo_msg += f", RT60: {geometry_analysis['rt60']}s"
                        if geometry_analysis.get('spl'):
                            geo_msg += f", SPL: {geometry_analysis['spl']}dBA"
                        if geometry_analysis.get('area'):
                            geo_msg += f", Area: {geometry_analysis['area']}m²"
                        self.chat_history.append(f"<span style='color:#9C27B0;'>{geo_msg}</span>")
                    
                    # Display standard summary
                    summary = result.get("summary", "No summary available")
                    self.chat_history.append(f"<span style='color:#E98801;'><b>AI:</b> {summary}</span>")
                    
                else:
                    # Standard result structure
                    self.display_standard_result(result)
            else:
                # Fallback for unexpected result types
                self.chat_history.append(f"<span style='color:#E98801;'><b>AI:</b> {str(result)}</span>")

        except Exception as e:
            self.chat_history.append(f"<span style='color:red;'>[Error]: {e}</span>")
            print(f"X Evaluation error: {e}")
            import traceback
            traceback.print_exc()

    def display_standard_result(self, result):
        """Display standard pipeline result in chat"""
        try:
            # Display comfort score
            comfort_score = result.get("comfort_score")
            if comfort_score is not None:
                self.chat_history.append(f"<span style='color:#4CAF50;'><b>🎯 Comfort Score:</b> {comfort_score}</span>")
            
            # Display source
            source = result.get("source")
            if source:
                self.chat_history.append(f"<span style='color:#2196F3;'><b>CHART Source:</b> {source}</span>")
            
            # Display compliance information
            compliance = result.get("compliance")
            if compliance:
                if isinstance(compliance, dict):
                    status = compliance.get("status", "Unknown")
                    reason = compliance.get("reason", "")
                    metrics = compliance.get("metrics", {})
                    details = compliance.get("details", [])
                    
                    # Status color based on compliance
                    if "OK" in status or "compliant" in status.lower():
                        status_color = "#4CAF50"
                    else:
                        status_color = "#F44336"
                    
                    self.chat_history.append(f"<span style='color:{status_color};'><b>📋 Compliance:</b> {status}</span>")
                    
                    # Display detailed compliance breakdown
                    if details:
                        details_text = "<b>CHART Compliance Breakdown:</b><br>"
                        for detail in details:
                            if "OK" in detail:
                                details_text += f"<span style='color:#4CAF50;'>• {detail}</span><br>"
                            else:
                                details_text += f"<span style='color:#F44336;'>• {detail}</span><br>"
                        self.chat_history.append(details_text)
                    
                    if reason:
                        self.chat_history.append(f"<span style='color:#FF9800;'><b>📝 Reason:</b> {reason}</span>")
                    
                    # Display metrics
                    if metrics:
                        metrics_text = "<b>CHART Metrics:</b><br>"
                        for metric, value in metrics.items():
                            if value is not None:
                                metrics_text += f"• {metric}: {value}<br>"
                        self.chat_history.append(f"<span style='color:#9C27B0;'>{metrics_text}</span>")
                else:
                    self.chat_history.append(f"<span style='color:#FF9800;'><b>📋 Compliance:</b> {compliance}</span>")
            
            # Display recommendations
            recommendations = result.get("recommendations")
            if recommendations:
                if isinstance(recommendations, dict):
                    rec_text = "<b>IDEA Recommendations:</b><br>"
                    for category, rec in recommendations.items():
                        if isinstance(rec, str):
                            rec_text += f"• <b>{category}:</b> {rec}<br>"
                        elif isinstance(rec, list):
                            for item in rec:
                                rec_text += f"• <b>{category}:</b> {item}<br>"
                    self.chat_history.append(f"<span style='color:#FF9800;'>{rec_text}</span>")
                elif isinstance(recommendations, list):
                    rec_text = "<b>IDEA Recommendations:</b><br>"
                    for rec in recommendations:
                        rec_text += f"• {rec}<br>"
                    self.chat_history.append(f"<span style='color:#FF9800;'>{rec_text}</span>")
                else:
                    self.chat_history.append(f"<span style='color:#FF9800;'><b>IDEA Recommendations:</b> {recommendations}</span>")
            
            # Display material upgrades
            best_materials = result.get("best_materials")
            if best_materials:
                materials_text = "<b>🏗️ Material Upgrades:</b><br>"
                for material_type, material_name in best_materials.items():
                    if material_name:
                        materials_text += f"• {material_type.replace('_', ' ').title()}: {material_name}<br>"
                self.chat_history.append(f"<span style='color:#9C27B0;'>{materials_text}</span>")
            
            # Display improved score
            improved_score = result.get("improved_score") or result.get("best_score")
            if improved_score:
                self.chat_history.append(f"<span style='color:#4CAF50;'><b>🚀 Improved Score:</b> {improved_score}</span>")
            
            # Display summary if available
            summary = result.get("summary")
            if summary:
                self.chat_history.append(f"<span style='color:#E98801;'><b>AI Summary:</b> {summary}</span>")
            
        except Exception as e:
            self.chat_history.append(f"<span style='color:red;'>X Error displaying result: {e}</span>")
            print(f"X Error in display_standard_result: {e}")

    def check_occ_selection(self):
        """Check for OCC shape selection and update InfoDock"""
        try:
            context = self.viewer._display.Context
            context.InitSelected()
            ais_shape = None

            while context.MoreSelected():
                ais_shape = context.SelectedInteractive()
                context.NextSelected()

            if ais_shape and ais_shape != getattr(self, 'last_selected', None):
                self.last_selected = ais_shape
                if ais_shape in self.viewer.shape_to_ifc:
                    ifc_elem = self.viewer.shape_to_ifc[ais_shape]
                    print(f"🎯 Selected element: {ifc_elem.is_a()} - {getattr(ifc_elem, 'Name', 'Unnamed')}")
                    self.show_ifc_panel(ifc_elem)
                else:
                    print("WARNING Selected shape not found in shape_to_ifc mapping")
                    self.show_ifc_panel(None)
            elif not ais_shape:
                if self.last_selected is not None:
                    print("SEARCH No element selected - clearing InfoDock")
                    self.last_selected = None
                    self.show_ifc_panel(None)
                    
        except Exception as e:
            print(f"X Error in check_occ_selection: {e}")

    def show_ifc_panel(self, ifc_elem):
        """Show enhanced IFC element information using the EnhancedInfoDock"""
        try:
            # Use the enhanced InfoDock's update_content method
            self.info_dock.update_content(ifc_elem)
            
            # If we have element data, trigger ML analysis automatically
            if ifc_elem and hasattr(self.info_dock, 'current_element_data') and self.info_dock.current_element_data:
                element_data = self.info_dock.current_element_data
                print(f"🤖 Element data available for ML: {element_data.get('type', 'Unknown')}")
                
                # Automatically run ML analysis for certain element types
                element_type = element_data.get('type', '')
                if element_type in ['IfcSpace', 'IfcWall', 'IfcSlab', 'IfcColumn', 'IfcBeam']:
                    print(f"🎯 Auto-triggering ML analysis for {element_type}")
                    self.run_ml_analysis_for_element(element_data)
                
                # Special handling for IfcSpace - show acoustic failure info
                if element_type == 'IfcSpace':
                    self.show_space_acoustic_info(ifc_elem)
                    
        except Exception as e:
            print(f"X Error showing IFC panel: {e}")
            self.info_dock.update_content(None)

    def show_space_acoustic_info(self, ifc_elem):
        """Show acoustic information for a selected space"""
        try:
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                return
            
            ifc_file = ifcopenshell.open(self.viewer.current_ifc_path)
            space_info = self.analyze_single_space(ifc_elem, ifc_file)
            
            if space_info:
                # Check for acoustic failures
                failures = self.check_acoustic_failures(space_info)
                severity = self.calculate_acoustic_severity(failures) if failures else "none"
                
                # Display acoustic information in chat
                self.chat_history.append(f"<span style='color:#9C27B0;'><b>MUSIC Space Acoustic Analysis: {space_info['name']}</b></span>")
                
                # Basic info
                self.chat_history.append(f"<span style='color:#2196F3;'>📏 Volume: {space_info['volume']:.1f} m³ | Area: {space_info['area']:.1f} m² | Height: {space_info['height']:.1f} m</span>")
                
                # Acoustic properties
                if space_info['acoustic_properties']:
                    self.chat_history.append(f"<span style='color:#4CAF50;'>MUSIC Acoustic Properties:</span>")
                    for prop, value in space_info['acoustic_properties'].items():
                        self.chat_history.append(f"<span style='color:#4CAF50;'>   • {prop}: {value}</span>")
                
                # Surrounding elements
                if space_info['surrounding_elements']:
                    self.chat_history.append(f"<span style='color:#FF9800;'>🏗️ Surrounding Elements:</span>")
                    for elem_type, count in space_info['surrounding_elements'].items():
                        self.chat_history.append(f"<span style='color:#FF9800;'>   • {elem_type}: {count}</span>")
                
                # Acoustic status
                if failures:
                    self.chat_history.append(f"<span style='color:#F44336;'>X Acoustic Status: {severity.upper()} RISK</span>")
                    self.chat_history.append(f"<span style='color:#F44336;'>🚨 Failures:</span>")
                    for failure in failures:
                        self.chat_history.append(f"<span style='color:#F44336;'>   • {failure}</span>")
                    
                    # Recommendations
                    recommendations = self.get_acoustic_recommendations(space_info, failures)
                    if recommendations:
                        self.chat_history.append(f"<span style='color:#4CAF50;'>IDEA Recommendations:</span>")
                        for rec in recommendations[:3]:  # Show top 3
                            self.chat_history.append(f"<span style='color:#4CAF50;'>   • {rec}</span>")
                else:
                    self.chat_history.append(f"<span style='color:#4CAF50;'>OK Acoustic Status: PASS - No issues detected</span>")
                
                # Add separator
                self.chat_history.append("<hr style='border-color:#FF9500; margin:10px 0;'>")
                
        except Exception as e:
            print(f"X Error showing space acoustic info: {e}")

    def run_ml_analysis_for_element(self, element_data):
        """Run ML analysis for a specific element and display results in chat"""
        try:
            if not element_data:
                self.chat_history.append("<span style='color:red;'>X No element data available for ML analysis</span>")
                return
            
            element_type = element_data.get('type', 'Unknown')
            element_name = element_data.get('name', 'Unnamed')
            
            self.chat_history.append(f"<b>🤖 Running ML Analysis for: {element_type} - {element_name}</b>")
            
            # Import and run ML analysis
            from scripts.core.geometry_ml_interface import geometry_ml_interface
            
            if geometry_ml_interface is None:
                self.chat_history.append("<span style='color:red;'>X ML interface not available</span>")
                return
            
            # Extract data for ML processing
            extracted_data = geometry_ml_interface.extract_element_data(element_data)
            if not extracted_data:
                self.chat_history.append("<span style='color:red;'>X Could not extract data for ML processing</span>")
                return
            
            self.chat_history.append(f"<span style='color:#2196F3;'>SEARCH Extracted data: {extracted_data.get('element_type', 'Unknown')}</span>")
            
            # Run comfort prediction
            comfort_result = geometry_ml_interface.predict_comfort_for_element(extracted_data)
            
            if "error" in comfort_result:
                self.chat_history.append(f"<span style='color:red;'>X ML Prediction Error: {comfort_result['error']}</span>")
            else:
                # Display comfort prediction results
                comfort_score = comfort_result.get('comfort_score', 'N/A')
                confidence = comfort_result.get('confidence', 0)
                self.chat_history.append(f"<span style='color:#4CAF50;'>🎯 Comfort Score: {comfort_score} (Confidence: {confidence:.1%})</span>")
                
                # Display key features used
                features_used = comfort_result.get('features_used', {})
                if features_used:
                    feature_text = "CHART Features used: "
                    if features_used.get('rt60_s'):
                        feature_text += f"RT60={features_used['rt60_s']}s, "
                    if features_used.get('spl_db'):
                        feature_text += f"SPL={features_used['spl_db']}dBA, "
                    if features_used.get('floor_height_m'):
                        feature_text += f"Height={features_used['floor_height_m']}m"
                    self.chat_history.append(f"<span style='color:#2196F3;'>{feature_text}</span>")
            
            # Generate and display recommendations
            recommendations = geometry_ml_interface.get_ml_recommendations(extracted_data)
            if recommendations:
                rec_text = "IDEA Recommendations:<br>"
                for rec in recommendations:
                    rec_text += f"• {rec}<br>"
                self.chat_history.append(f"<span style='color:#FF9800;'>{rec_text}</span>")
            else:
                self.chat_history.append("<span style='color:#FF9800;'>IDEA No specific recommendations available</span>")
            
            # Add separator
            self.chat_history.append("<hr style='border-color:#FF9500; margin:10px 0;'>")
            
        except Exception as e:
            error_msg = f"X ML analysis failed: {str(e)}"
            self.chat_history.append(f"<span style='color:red;'>{error_msg}</span>")
            print(error_msg)
            import traceback
            traceback.print_exc()

    def reset_element_colors(self):
        """Reset colors to original element type colors"""
        self.viewer.reset_to_element_colors()
        self.apartment_btn.setText("Toggle Apartment Overlay")
        self.apartment_btn.setStyleSheet(btn_style)
        self.chat_history.append("OK Reset to element type colors")

    def debug_ifc_file(self):
        """Debug IFC file contents to help diagnose loading issues"""
        if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
            QMessageBox.information(self, "Debug", "No IFC file loaded. Please load an IFC file first.")
            return
            
        try:
            element_counts = self.viewer.analyze_ifc_file(self.viewer.current_ifc_path)
            
            # Create detailed debug report
            debug_text = f"SEARCH IFC Debug Report: {os.path.basename(self.viewer.current_ifc_path)}\n"
            debug_text += "=" * 50 + "\n"
            
            if element_counts:
                debug_text += "CHART Element Analysis:\n"
                for elem_type, count in sorted(element_counts.items()):
                    debug_text += f"  • {elem_type}: {count}\n"
                
                # Check for geometric elements
                geometric_types = [
                    "IfcWall", "IfcSlab", "IfcBeam", "IfcColumn", "IfcDoor", "IfcWindow",
                    "IfcRoof", "IfcStair", "IfcFurniture", "IfcSanitaryTerminal",
                    "IfcRailing", "IfcCovering", "IfcPlate", "IfcMember", "IfcFooting",
                    "IfcPile", "IfcBuildingElementProxy", "IfcDistributionElement",
                    "IfcFlowTerminal", "IfcFlowSegment", "IfcFlowFitting"
                ]
                
                geometric_count = sum(element_counts.get(t, 0) for t in geometric_types)
                debug_text += f"\n🎯 Geometric elements: {geometric_count}\n"
                
                if geometric_count == 0:
                    debug_text += "WARNING No geometric elements found!\n"
                    debug_text += "IDEA This IFC file may not contain loadable geometry.\n"
                    debug_text += "IDEA Try a different IFC file with walls, slabs, or other building elements.\n"
                else:
                    debug_text += "OK Geometric elements found - should load successfully.\n"
                
                # Check loaded elements
                loaded_count = len(self.viewer.shape_to_ifc)
                debug_text += f"\nPACKAGE Currently loaded: {loaded_count} elements\n"
                
                if loaded_count == 0 and geometric_count > 0:
                    debug_text += "WARNING Elements found but not loaded - there may be a loading issue.\n"
                elif loaded_count > 0:
                    debug_text += "OK Elements loaded successfully.\n"
            
            # Show in chat
            self.chat_history.append(debug_text)
            
            # Also show in a dialog for better visibility
            QMessageBox.information(self, "IFC Debug Report", debug_text)
            
        except Exception as e:
            error_msg = f"X Debug error: {str(e)}"
            self.chat_history.append(error_msg)
            QMessageBox.critical(self, "Debug Error", error_msg)

    def analyze_ifc_file(self, path):
        """Analyze IFC file and return element counts"""
        try:
            ifc_file = ifcopenshell.open(path)
            
            # Count elements by type
            element_counts = {}
            for product in ifc_file.by_type("IfcProduct"):
                elem_type = product.is_a()
                element_counts[elem_type] = element_counts.get(elem_type, 0) + 1
            
            return element_counts
            
        except Exception as e:
            print(f"X Error analyzing IFC file: {e}")
            return None

    def debug_ml_integration(self):
        """Debug ML integration and show current status"""
        try:
            self.chat_history.append("<b>SEARCH ML Integration Debug</b>")
            
            # Check if models are available
            model_files = [
                "model/ecoform_xgb_comfort_model1.pkl",
                "model/ecoform_xgb_comfort_focused2.pkl",
                "model/ecoform_acoustic_comfort_model.pkl"
            ]
            
            missing_models = []
            for model_file in model_files:
                if not os.path.exists(model_file):
                    missing_models.append(model_file)
            
            if missing_models:
                self.chat_history.append(f"<span style='color:red;'>X Missing model files: {missing_models}</span>")
            else:
                self.chat_history.append("<span style='color:green;'>OK All model files found</span>")
            
            # Test ML prediction
            test_data = {
                'element_type': 'IfcWall',
                'apartment_type': '2B',
                'floor_level': 2,
                'area': 25.0,
                'rt60': 0.8,
                'spl': 45.0,
                'materials': ['Concrete'],
                'location': {'x': 0, 'y': 0, 'z': 0}
            }
            
            try:
                result = self.test_ml_prediction(test_data)
                self.chat_history.append(f"<span style='color:green;'>OK ML Test Result: {result}</span>")
            except Exception as e:
                self.chat_history.append(f"<span style='color:red;'>X ML Test Failed: {e}</span>")
                
        except Exception as e:
            self.chat_history.append(f"<span style='color:red;'>X Debug error: {e}</span>")
            print(f"X Debug error: {e}")

    def debug_ifc_analysis(self):
        """Debug IFC analysis to see what data is available"""
        try:
            self.chat_history.append("<b>SEARCH IFC Analysis Debug Report</b>")
            
            # Check if IFC file is loaded
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                self.chat_history.append("X No IFC file loaded")
                return
            
            self.chat_history.append(f"📁 IFC File: {os.path.basename(self.viewer.current_ifc_path)}")
            
            # Get comprehensive analysis
            analysis = self.get_comprehensive_ifc_analysis()
            self.chat_history.append(f"CHART Analysis Results:")
            
            # Split analysis into lines and display
            lines = analysis.split('\n')
            for line in lines:
                if line.strip():
                    self.chat_history.append(f"   {line}")
            
            # Test the analysis with a sample question
            self.chat_history.append(f"\n🧪 Testing with sample question: 'How many IfcSpace are in the building?'")
            
            # Simulate the chatbot response
            try:
                summary = f"""
IFC Model Analysis:
{analysis}

Current Loaded Elements:
- Total loaded: {len(self.viewer.shape_to_ifc)} elements
- Loaded types: {', '.join(set(elem.is_a() for elem in self.viewer.shape_to_ifc.values()))}

Note: Some IFC elements (like IfcSpace) may not have geometric representation and won't appear in the 3D viewer, but they exist in the IFC data structure.
"""

                prompt = f"How many IfcSpace are in the building?\n\n{summary}"

                from server.config import client, completion_model
                response = client.chat.completions.create(
                    model=completion_model,
                    messages=[
                        {"role": "system", "content": "You are an architectural acoustics advisor with expertise in IFC (Industry Foundation Classes) building models. You can analyze building elements, spaces, and provide acoustic insights. When asked about IFC elements, provide specific counts and details from the data provided."},
                        {"role": "user", "content": prompt}
                    ]
                )

                answer = response.choices[0].message.content.strip()
                self.chat_history.append(f"🤖 AI Response: {answer}")
                
            except Exception as e:
                self.chat_history.append(f"X AI test failed: {e}")
            
        except Exception as e:
            self.chat_history.append(f"X IFC Analysis debug error: {e}")
            import traceback
            traceback.print_exc()

    def test_ml_prediction(self, element_data):
        """Test ML prediction with sample data"""
        try:
            from scripts.core.geometry_ml_interface import GeometryMLInterface
            ml_interface = GeometryMLInterface()
            
            # Test prediction
            prediction = ml_interface.predict_comfort(element_data)
            return f"ML Prediction: {prediction}"
        except Exception as e:
            return f"ML Test Error: {e}"

    def test_rag_analysis(self):
        """RAG functionality removed - using LLM calls instead"""
        self.chat_history.append("<span style='color:#FF9800;'>WARNING RAG functionality has been removed. Using LLM calls for better performance.</span>")
        return

    # Enhanced Rendering Methods
    def toggle_enhanced_rendering(self):
        """Toggle enhanced rendering on/off"""
        if not self.viewer.enhanced_rendering_enabled:
            self.viewer.enable_enhanced_rendering()
            self.enhanced_btn.setText("Disable Enhanced Rendering")
            self.enhanced_btn.setStyleSheet(f"""
                QPushButton {{
                    background: #4CAF50;
                    color: white;
                    border-radius: 10px;
                    padding: 10px 20px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background: #45a049;
                }}
            """)
            print("OK Enhanced rendering enabled")
        else:
            self.viewer.enhanced_rendering_enabled = False
            self.enhanced_btn.setText("Enable Enhanced Rendering")
            self.enhanced_btn.setStyleSheet(btn_style)
            print("OK Enhanced rendering disabled")

    def toggle_apartment_overlay(self):
        """Toggle apartment overlay on/off"""
        if not self.viewer.apartment_overlay_enabled:
            stats = self.viewer.create_apartment_overlay()
            if stats:
                self.apartment_btn.setText("Disable Apartment Overlay")
                self.apartment_btn.setStyleSheet(f"""
                    QPushButton {{
                        background: #2196F3;
                        color: white;
                        border-radius: 10px;
                        padding: 10px 20px;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background: #1976D2;
                    }}
                """)
                # Set apartment overlay as active in InfoDock and update content
                if hasattr(self, 'info_dock') and self.info_dock:
                    self.info_dock.set_apartment_overlay_active(True)
                    self.info_dock.update_content(None)  # Show legend only
                
                # Show statistics in chat
                stats_text = f"BUILDING Apartment Overlay Applied:\n"
                stats_text += f"• Total Apartments: {stats['total_apartments']}\n"
                stats_text += f"• Total Elements: {stats['total_elements']}\n"
                for apt_type, count in stats['apartment_counts'].items():
                    stats_text += f"• {apt_type}: {count} elements\n"
                self.chat_history.append(stats_text)
            else:
                QMessageBox.warning(self, "Warning", "No apartment spaces found in the IFC file.")
        else:
            self.viewer.reset_to_element_colors()
            self.apartment_btn.setText("Toggle Apartment Overlay")
            self.apartment_btn.setStyleSheet(btn_style)
            # Set apartment overlay as inactive in InfoDock and update content
            if hasattr(self, 'info_dock') and self.info_dock:
                self.info_dock.set_apartment_overlay_active(False)
                self.info_dock.update_content(None)
            self.chat_history.append("OK Reset to element type colors")

    def reset_element_colors(self):
        """Reset colors to original element type colors"""
        self.viewer.reset_to_element_colors()
        self.apartment_btn.setText("Toggle Apartment Overlay")
        self.apartment_btn.setStyleSheet(btn_style)
        self.chat_history.append("OK Reset to element type colors")

    def calculate_acoustic_severity(self, failures):
        """Calculate severity level based on failures"""
        try:
            severity_score = 0
            
            for failure in failures:
                if "excessive" in failure.lower() or "high" in failure.lower():
                    severity_score += 3
                elif "insufficient" in failure.lower() or "low" in failure.lower():
                    severity_score += 2
                elif "too many" in failure.lower():
                    severity_score += 2
                else:
                    severity_score += 1
            
            if severity_score >= 6:
                return "critical"
            elif severity_score >= 4:
                return "high"
            elif severity_score >= 2:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            return "medium"

    def get_severity_color(self, severity):
        """Get color based on severity level - simplified: green for pass, red for fail"""
        if severity == "low":
            return (0.0, 1.0, 0.0)  # Green for pass
        else:
            return (1.0, 0.0, 0.0)  # Red for all failures (medium, high, critical)

    def get_acoustic_recommendations(self, space_info, failures):
        """Get specific recommendations for improving acoustic performance - USING UI10 LOGIC"""
        recommendations = []
        
        try:
            volume = space_info.get('volume', 0)
            area = space_info.get('area', 0)
            height = space_info.get('height', 0)
            surrounding = space_info.get('surrounding_elements', {})
            
            # Volume-based recommendations
            if volume > 2000:
                recommendations.append("Install acoustic panels and diffusers to control reverberation")
                recommendations.append("Consider suspended acoustic ceiling")
            
            if volume < 30:
                recommendations.append("Add acoustic absorption materials to walls and ceiling")
                recommendations.append("Consider sound masking system")
            
            # Area-based recommendations
            if area > 500:
                recommendations.append("Divide large space with acoustic partitions")
                recommendations.append("Install carpet or acoustic flooring")
            
            # Height-based recommendations
            if height > 5:
                recommendations.append("Install suspended acoustic ceiling at 2.7-3.0m height")
                recommendations.append("Add wall-mounted acoustic panels")
            
            if height < 2.4:
                recommendations.append("Increase ceiling height if possible")
                recommendations.append("Use low-profile acoustic treatments")
            
            # Element-based recommendations
            wall_count = surrounding.get('IfcWall', 0)
            door_count = surrounding.get('IfcDoor', 0)
            window_count = surrounding.get('IfcWindow', 0)
            
            # Handle counts properly
            if isinstance(wall_count, (list, tuple)):
                wall_count = len(wall_count)
            elif not isinstance(wall_count, (int, float)):
                wall_count = 0
                
            if isinstance(door_count, (list, tuple)):
                door_count = len(door_count)
            elif not isinstance(door_count, (int, float)):
                door_count = 0
                
            if isinstance(window_count, (list, tuple)):
                window_count = len(window_count)
            elif not isinstance(window_count, (int, float)):
                window_count = 0
            
            if wall_count < 3:
                recommendations.append("Add acoustic walls or partitions")
                recommendations.append("Install sound-absorbing wall panels")
            
            if door_count > 5:
                recommendations.append("Replace standard doors with acoustic doors")
                recommendations.append("Install door seals and sweeps")
            
            if window_count > 4:
                recommendations.append("Install double-glazed acoustic windows")
                recommendations.append("Add acoustic window treatments")
            
            # General recommendations
            if not space_info.get('acoustic_properties'):
                recommendations.append("Define acoustic properties for the space")
                recommendations.append("Conduct acoustic measurements and analysis")
            
            return recommendations[:5]  # Limit to 5 most important
            
        except Exception as e:
            return [f"Error generating recommendations: {e}"]

    def reset_acoustic_failure_overlay(self):
        """Reset the acoustic failure overlay colors"""
        try:
            # Reset all space colors to original
            for ais_shape, ifc_elem in self.viewer.shape_to_ifc.items():
                if ifc_elem.is_a() == "IfcSpace":
                    # Reset to original space color
                    color = self.viewer.color_map.get("IfcSpace", self.viewer.default_color)
                    color_obj = Quantity_Color(color[0], color[1], color[2], Quantity_TOC_RGB)
                    self.viewer._display.Context.SetColor(ais_shape, color_obj, False)
                    self.viewer._display.Context.SetTransparency(ais_shape, 0.7, False)
            
            # Update display
            self.viewer._display.Context.UpdateCurrentViewer()
            self.viewer._display.Repaint()
            
            self.chat_history.append("<span style='color:#4CAF50;'>OK Acoustic failure overlay reset</span>")
            
        except Exception as e:
            print(f"X Error resetting overlay: {e}")

    def debug_acoustic_analysis(self):
        """Debug acoustic analysis to help diagnose issues"""
        try:
            self.chat_history.append("<b>SEARCH Acoustic Analysis Debug Report</b>")
            
            # Check if IFC file is loaded
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                self.chat_history.append("<span style='color:red;'>X No IFC file loaded</span>")
                return
            
            self.chat_history.append(f"<span style='color:#2196F3;'>📁 IFC File: {os.path.basename(self.viewer.current_ifc_path)}</span>")
            
            # Check IFC file structure
            try:
                ifc_file = ifcopenshell.open(self.viewer.current_ifc_path)
                spaces = ifc_file.by_type("IfcSpace")
                self.chat_history.append(f"<span style='color:#4CAF50;'>OK Found {len(spaces)} IfcSpace elements</span>")
                
                if spaces:
                    # Analyze first few spaces
                    self.chat_history.append(f"<span style='color:#FF9500;'>SEARCH Analyzing first 3 spaces:</span>")
                    
                    for i, space in enumerate(spaces[:3]):
                        try:
                            space_info = self.analyze_single_space(space, ifc_file)
                            if space_info:
                                self.chat_history.append(f"<span style='color:#9C27B0;'>📍 Space {i+1}: {space_info['name']}</span>")
                                self.chat_history.append(f"<span style='color:#9C27B0;'>   • Volume: {space_info['volume']:.1f} m³</span>")
                                self.chat_history.append(f"<span style='color:#9C27B0;'>   • Area: {space_info['area']:.1f} m²</span>")
                                self.chat_history.append(f"<span style='color:#9C27B0;'>   • Height: {space_info['height']:.1f} m</span>")
                                
                                # Check for failures
                                failures = self.check_acoustic_failures(space_info)
                                if failures:
                                    self.chat_history.append(f"<span style='color:#F44336;'>   • Failures: {len(failures)} found</span>")
                                    for failure in failures[:2]:  # Show first 2
                                        self.chat_history.append(f"<span style='color:#F44336;'>     - {failure}</span>")
                                else:
                                    self.chat_history.append(f"<span style='color:#4CAF50;'>   • Status: No failures detected</span>")
                            else:
                                self.chat_history.append(f"<span style='color:#FF9800;'>   • Could not analyze space {i+1}</span>")
                        except Exception as e:
                            self.chat_history.append(f"<span style='color:red;'>   • Error analyzing space {i+1}: {str(e)[:50]}...</span>")
                    
                    # Look for specific spaces that might have issues (like LVL4_2B_39)
                    self.chat_history.append(f"<span style='color:#FF9500;'>SEARCH Looking for potential problematic spaces:</span>")
                    
                    problematic_spaces = []
                    for space in spaces:
                        space_name = getattr(space, 'Name', '')
                        if space_name is None:
                            space_name = ''
                        if any(indicator in space_name.upper() for indicator in ['LVL4', 'LVL5', 'LVL6', 'ROOF', 'MECH']):
                            problematic_spaces.append(space_name)
                    
                    if problematic_spaces:
                        self.chat_history.append(f"<span style='color:#FF9800;'>WARNING Found {len(problematic_spaces)} potentially problematic spaces:</span>")
                        for space_name in problematic_spaces[:5]:  # Show first 5
                            self.chat_history.append(f"<span style='color:#FF9800;'>   • {space_name}</span>")
                        
                        # Analyze a specific problematic space if found
                        if 'LVL4_2B_39' in problematic_spaces:
                            self.chat_history.append(f"<span style='color:#F44336;'>🎯 Analyzing specific space: LVL4_2B_39</span>")
                            specific_analysis = self.analyze_specific_space('LVL4_2B_39')
                            if specific_analysis and "Error" not in specific_analysis:
                                # Show key parts of the analysis
                                lines = specific_analysis.split('\n')
                                for line in lines[:10]:  # Show first 10 lines
                                    if line.strip():
                                        self.chat_history.append(f"<span style='color:#F44336;'>{line}</span>")
                                if len(lines) > 10:
                                    self.chat_history.append(f"<span style='color:#F44336;'>   ... (analysis continues)</span>")
                            else:
                                self.chat_history.append(f"<span style='color:red;'>X Could not analyze LVL4_2B_39: {specific_analysis}</span>")
                    else:
                        self.chat_history.append(f"<span style='color:#4CAF50;'>OK No obviously problematic spaces found</span>")
                        
                else:
                    self.chat_history.append("<span style='color:#FF9800;'>WARNING No IfcSpace elements found in IFC file</span>")
                    
            except Exception as e:
                self.chat_history.append(f"<span style='color:red;'>X Error reading IFC file: {e}</span>")
            
            # Show analysis capabilities
            self.chat_history.append(f"<span style='color:#9C27B0;'><b>CHART Analysis Capabilities:</b></span>")
            self.chat_history.append(f"<span style='color:#9C27B0;'>• Smart space selection (prioritizes problematic spaces)</span>")
            self.chat_history.append(f"<span style='color:#9C27B0;'>• Volume, area, and height analysis</span>")
            self.chat_history.append(f"<span style='color:#9C27B0;'>• Surrounding element assessment</span>")
            self.chat_history.append(f"<span style='color:#9C27B0;'>• Acoustic property evaluation</span>")
            self.chat_history.append(f"<span style='color:#9C27B0;'>• Specific space analysis (e.g., LVL4_2B_39)</span>")
            
        except Exception as e:
            self.chat_history.append(f"<span style='color:red;'>X Debug error: {e}</span>")
            print(f"X Debug error: {e}")
            import traceback
            traceback.print_exc()

    def analyze_specific_space(self, space_name):
        """Analyze a specific space by name"""
        try:
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                return f"Error: No IFC file loaded"
            
            ifc_file = ifcopenshell.open(self.viewer.current_ifc_path)
            spaces = ifc_file.by_type("IfcSpace")
            
            # Find the specific space
            target_space = None
            for space in spaces:
                current_name = getattr(space, 'Name', '')
                if current_name is None:
                    current_name = ''
                if current_name == space_name:
                    target_space = space
                    break
            
            if not target_space:
                return f"Error: Space '{space_name}' not found in IFC file"
            
            # Analyze the space
            space_info = self.analyze_single_space(target_space, ifc_file)
            if not space_info:
                return f"Error: Could not analyze space '{space_name}'"
            
            # Check for failures
            failures = self.check_acoustic_failures(space_info)
            
            # Build analysis report
            report = []
            report.append(f"SEARCH Detailed Analysis: {space_name}")
            report.append("=" * 50)
            report.append(f"📋 Basic Information:")
            report.append(f"   • Global ID: {space_info['global_id']}")
            report.append(f"   • Volume: {space_info['volume']:.1f} m³")
            report.append(f"   • Area: {space_info['area']:.1f} m²")
            report.append(f"   • Height: {space_info['height']:.1f} m")
            
            if space_info.get('apartment_type'):
                report.append(f"   • Apartment Type: {space_info['apartment_type']}")
            
            # Acoustic properties
            if space_info['acoustic_properties']:
                report.append(f"\nMUSIC Acoustic Properties:")
                for prop, value in space_info['acoustic_properties'].items():
                    report.append(f"   • {prop}: {value}")
            else:
                report.append(f"\nMUSIC Acoustic Properties: None defined")
            
            # Surrounding elements
            if space_info['surrounding_elements']:
                report.append(f"\n🏗️ Surrounding Elements:")
                for elem_type, count in space_info['surrounding_elements'].items():
                    report.append(f"   • {elem_type}: {count}")
            else:
                report.append(f"\n🏗️ Surrounding Elements: None found")
            
            # Acoustic risk assessment
            report.append(f"\nWARNING Acoustic Risk Assessment:")
            report.append(f"   • Risk Level: {space_info['acoustic_risk']}")
            
            # Failure analysis
            if failures:
                report.append(f"\nX Acoustic Failures Detected:")
                for failure in failures:
                    report.append(f"   • {failure}")
                
                # Recommendations
                recommendations = self.get_acoustic_recommendations(space_info, failures)
                if recommendations:
                    report.append(f"\nIDEA Recommendations:")
                    for rec in recommendations:
                        report.append(f"   • {rec}")
            else:
                report.append(f"\nOK No acoustic failures detected")
                report.append(f"   This space appears to meet basic acoustic requirements")
            
            return "\n".join(report)
            
        except Exception as e:
            return f"Error analyzing space '{space_name}': {e}"

    def analyze_specific_space_ui(self, space_name):
        """Analyze a specific space and display results in UI"""
        try:
            self.chat_history.append(f"<b>SEARCH Analyzing Specific Space: {space_name}</b>")
            
            # Run the analysis
            analysis_result = self.analyze_specific_space(space_name)
            
            if analysis_result and "Error" not in analysis_result:
                # Display the analysis results
                lines = analysis_result.split('\n')
                for line in lines:
                    if line.strip():
                        # Color code different sections
                        if "FAILURES" in line or "X" in line:
                            self.chat_history.append(f"<span style='color:#F44336;'>{line}</span>")
                        elif "RECOMMENDATIONS" in line or "IDEA" in line:
                            self.chat_history.append(f"<span style='color:#4CAF50;'>{line}</span>")
                        elif "Basic Information" in line or "📋" in line:
                            self.chat_history.append(f"<span style='color:#2196F3;'>{line}</span>")
                        elif "Acoustic Properties" in line or "MUSIC" in line:
                            self.chat_history.append(f"<span style='color:#9C27B0;'>{line}</span>")
                        elif "Surrounding Elements" in line or "🏗️" in line:
                            self.chat_history.append(f"<span style='color:#FF9800;'>{line}</span>")
                        elif "Risk Assessment" in line or "WARNING" in line:
                            self.chat_history.append(f"<span style='color:#FF5722;'>{line}</span>")
                        elif "No acoustic failures" in line or "OK" in line:
                            self.chat_history.append(f"<span style='color:#4CAF50;'>{line}</span>")
                        else:
                            self.chat_history.append(f"<span style='color:#FDF6F6;'>{line}</span>")
                
                # Add separator
                self.chat_history.append("<hr style='border-color:#FF9500; margin:10px 0;'>")
                
            else:
                self.chat_history.append(f"<span style='color:red;'>X {analysis_result}</span>")
                
        except Exception as e:
            self.chat_history.append(f"<span style='color:red;'>X Error analyzing space {space_name}: {e}</span>")
            print(f"X Error in analyze_specific_space_ui: {e}")
            import traceback
            traceback.print_exc()

    def check_acoustic_failures(self, space_info):
        """Check for specific acoustic failures in a space - USING UI10 LENIENT LOGIC"""
        failures = []
        
        try:
            # Import compliance thresholds and ML inference
            from utils.reference_data import DAY_RANGES, NIGHT_RANGES, RT60_target, RT60_max_dev
            from utils.infer_from_inputs import infer_features
            import json
            
            # Load enhanced compliance guidance as primary source
            try:
                with open('knowledge/compliance_guidance.json', 'r') as f:
                    compliance_guidance = json.load(f)
                print("[DEBUG] Successfully loaded compliance guidance JSON")
            except Exception as e:
                print(f"[DEBUG] Failed to load compliance guidance JSON: {e}")
                compliance_guidance = {}
            
            # Get space properties
            volume = space_info.get('volume', 0)
            area = space_info.get('area', 0)
            height = space_info.get('height', 0)
            space_name = space_info.get('name', '')
            apartment_type = space_info.get('apartment_type', '2B')  # Default to 2B
            zone = space_info.get('zone', 'HD-Urban-V1')  # Default zone
            time_period = space_info.get('time_period', 'day')
            
            # Use ML inference to get realistic acoustic values
            try:
                # Prepare inputs for ML inference
                wall_material = space_info.get('wall_material', 'Concrete Block (Painted)')
                window_material = space_info.get('window_material', 'Double Pane Glass')
                floor_level = space_info.get('floor_level', 1)
                
                # Call ML inference to get realistic acoustic values
                features, tier = infer_features(
                    apartment_type=apartment_type,
                    zone=zone,
                    element='wall',
                    element_material=wall_material,
                    floor_level=floor_level,
                    wall_material=wall_material,
                    window_material=window_material,
                    time_period=time_period
                )
                
                print(f"[DEBUG] ML inference tier: {tier}")
                print(f"[DEBUG] Generated features: {features}")
                
                # Extract realistic acoustic values from ML inference
                laeq_value = features.get('laeq_db', 45.0)  # Default from ML
                rt60_value = features.get('rt60_s', 0.4)    # Default from ML
                spl_value = features.get('spl_db', 50.0)    # Default from ML
                comfort_score = features.get('comfort_index_float', 0.7)  # Default from ML
                
                # Update space_info with realistic values
                space_info['acoustic_properties'] = {
                    'LAeq': laeq_value,
                    'RT60': rt60_value,
                    'SPL': spl_value,
                    'comfort_score': comfort_score,
                    'absorption_coefficient': features.get('absorption_coefficient_by_area_m', 0.1)
                }
                
            except Exception as e:
                print(f"[DEBUG] ML inference failed: {e}")
                # Fallback to default values
                laeq_value = 45.0
                rt60_value = 0.4
                spl_value = 50.0
                comfort_score = 0.7
                space_info['acoustic_properties'] = {
                    'LAeq': laeq_value,
                    'RT60': rt60_value,
                    'SPL': spl_value,
                    'comfort_score': comfort_score
                }
            
            # === UI10 LENIENT LOGIC - Much more forgiving thresholds ===
            
            # Volume-based failures (UI10 logic - much more lenient)
            if isinstance(volume, (int, float)):
                if volume > 2000:  # UI10: 2000 vs UI12: 1500
                    failures.append("Excessive volume (>2000 m³) - may cause poor speech intelligibility")
                elif volume < 30:   # UI10: 30 vs UI12: 40
                    failures.append("Very small volume (<30 m³) - may cause sound pressure issues")
            
            # Area-based failures (UI10 logic - much more lenient)
            if isinstance(area, (int, float)):
                if area > 500:  # UI10: 500 vs UI12: 300
                    failures.append("Large floor area (>500 m²) - may require acoustic treatment")
            
            # Height-based failures (UI10 logic - much more lenient)
            if isinstance(height, (int, float)):
                if height > 5:    # UI10: 5 vs UI12: 4
                    failures.append("High ceiling (>5m) - may cause excessive reverberation")
                elif height < 2.4: # UI10: 2.4 vs UI12: 2.5
                    failures.append("Low ceiling (<2.4m) - may cause sound pressure issues")
            
            # Surrounding elements failures (UI10 logic - much more lenient)
            surrounding = space_info.get('surrounding_elements', {})
            
            # Handle wall count - ensure it's an integer
            wall_count = surrounding.get('IfcWall', 0)
            if isinstance(wall_count, (list, tuple)):
                wall_count = len(wall_count)
            elif not isinstance(wall_count, (int, float)):
                wall_count = 0
            
            # Handle door count
            door_count = surrounding.get('IfcDoor', 0)
            if isinstance(door_count, (list, tuple)):
                door_count = len(door_count)
            elif not isinstance(door_count, (int, float)):
                door_count = 0
            
            # Handle window count
            window_count = surrounding.get('IfcWindow', 0)
            if isinstance(window_count, (list, tuple)):
                window_count = len(window_count)
            elif not isinstance(window_count, (int, float)):
                window_count = 0
            
            # Check counts (UI10 logic - much more lenient)
            if wall_count < 3:    # UI10: 3 vs UI12: 4
                failures.append("Insufficient walls (<3) - poor acoustic isolation")
            if door_count > 5:    # UI10: 5 vs UI12: 3
                failures.append("Too many doors (>5) - excessive sound leakage")
            if window_count > 4:  # UI10: 4 vs UI12: 2
                failures.append("Too many windows (>4) - poor sound insulation")
            
            # Acoustic properties failures (UI10 logic - much more lenient)
            acoustic_props = space_info.get('acoustic_properties', {})
            if not acoustic_props:
                failures.append("No acoustic properties defined - cannot assess performance")
            else:
                for prop_name, value in acoustic_props.items():
                    if 'rt60' in prop_name.lower() and value:
                        try:
                            rt60_val = float(value)
                            if rt60_val > 3.0:  # UI10: 3.0 vs UI12: 2.5
                                failures.append(f"Excessive RT60 ({rt60_val}s) - poor acoustic performance")
                            elif rt60_val < 0.2:  # UI10: 0.2 vs UI12: 0.1
                                failures.append(f"Very low RT60 ({rt60_val}s) - space may be too dead")
                        except:
                            pass
                    
                    if 'spl' in prop_name.lower() and value:
                        try:
                            spl_val = float(value)
                            if spl_val > 70:  # UI10: 70 vs UI12: various lower thresholds
                                failures.append(f"High SPL ({spl_val} dBA) - excessive noise levels")
                        except:
                            pass
            
            # === SUMMARY ===
            if not failures:
                print(f"OK Space {space_name} PASSES all acoustic criteria (UI10 logic)")
            else:
                print(f"X Space {space_name} has {len(failures)} acoustic failures")
            
            return failures
            
        except Exception as e:
            print(f"X Error in acoustic failure checking: {e}")
            return [f"Error in acoustic analysis: {str(e)}"]

    def detect_apartment_type(self, name):
        # Enhanced apartment type detection with comprehensive pattern matching
        import re
        
        # Handle None or empty names
        if not name:
            print(f"[DEBUG] Empty name provided to detect_apartment_type")
            return None
        
        try:
            # Convert to string and uppercase for consistent matching
            name_str = str(name).upper()
            print(f"[DEBUG] Analyzing name: '{name}' -> '{name_str}'")
            
            # Comprehensive patterns to match apartment types
            patterns = [
                # Standard patterns
                r'([123])[_ ]?B(ED)?',  # 1B, 2B, 3B, 1BED, 2BED, 3BED
                r'LVL\d+_([123])B_\d+',  # LVL1_1B_001, LVL2_2B_002, etc.
                r'APT_([123])B',  # APT_1B, APT_2B, APT_3B
                r'APARTMENT_([123])B',  # APARTMENT_1B, etc.
                r'([123])BED',  # 1BED, 2BED, 3BED
                r'([123])_BED',  # 1_BED, 2_BED, 3_BED
                r'BEDROOM_([123])',  # BEDROOM_1, BEDROOM_2, BEDROOM_3
                r'([123])BR',  # 1BR, 2BR, 3BR
                r'([123])_BR',  # 1_BR, 2_BR, 3_BR
                
                # Additional patterns for various naming conventions
                r'([123])BEDROOM',  # 1BEDROOM, 2BEDROOM, 3BEDROOM
                r'([123])_BEDROOM',  # 1_BEDROOM, 2_BEDROOM, 3_BEDROOM
                r'UNIT_([123])B',  # UNIT_1B, UNIT_2B, UNIT_3B
                r'FLAT_([123])B',  # FLAT_1B, FLAT_2B, FLAT_3B
                r'SUITE_([123])B',  # SUITE_1B, SUITE_2B, SUITE_3B
                r'ROOM_([123])B',  # ROOM_1B, ROOM_2B, ROOM_3B
                r'SPACE_([123])B',  # SPACE_1B, SPACE_2B, SPACE_3B
                r'ZONE_([123])B',  # ZONE_1B, ZONE_2B, ZONE_3B
                
                # Patterns with numbers and letters
                r'([123])B[A-Z]?',  # 1BA, 2BA, 3BA, etc.
                r'([123])BED[A-Z]?',  # 1BEDA, 2BEDA, 3BEDA, etc.
                
                # Patterns with underscores and dashes
                r'([123])-B',  # 1-B, 2-B, 3-B
                r'([123])-BED',  # 1-BED, 2-BED, 3-BED
                r'([123])-BR',  # 1-BR, 2-BR, 3-BR
                
                # Patterns with "TYPE" or "TYP"
                r'TYPE_([123])B',  # TYPE_1B, TYPE_2B, TYPE_3B
                r'TYP_([123])B',  # TYP_1B, TYP_2B, TYP_3B
                
                # Patterns with level information
                r'LEVEL\d+_([123])B',  # LEVEL1_1B, LEVEL2_2B, etc.
                r'FLOOR\d+_([123])B',  # FLOOR1_1B, FLOOR2_2B, etc.
                
                # Generic patterns that might contain apartment info
                r'([123])B[A-Z0-9_]*$',  # 1B followed by any characters
                r'[A-Z0-9_]*([123])B[A-Z0-9_]*',  # 1B, 2B, 3B anywhere in string
            ]
            
            for i, pattern in enumerate(patterns):
                match = re.search(pattern, name_str)
                if match:
                    bedroom_count = match.group(1)
                    result = f"{bedroom_count}B"
                    print(f"[DEBUG] Pattern {i+1} matched: '{pattern}' -> '{result}'")
                    return result
            
            # If no pattern matches, try to extract any number that might indicate bedrooms
            number_match = re.search(r'([123])\D', name_str)
            if number_match:
                bedroom_count = number_match.group(1)
                result = f"{bedroom_count}B"
                print(f"[DEBUG] Fallback number match: '{bedroom_count}' -> '{result}'")
                return result
            
            print(f"[DEBUG] No apartment type pattern matched for '{name_str}'")
            return None
            
        except Exception as e:
            print(f"[DEBUG] Error in detect_apartment_type for '{name}': {e}")
            return None

    def get_ifc_spaces_summary(self):
        """Return a summary of all spaces and their key properties from the loaded IFC model."""
        try:
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                return "No IFC file loaded."
            ifc_file = ifcopenshell.open(self.viewer.current_ifc_path)
            spaces = ifc_file.by_type("IfcSpace")
            summary = []
            for space in spaces:
                info = self.analyze_single_space(space, ifc_file)
                if info:
                    summary.append({
                        'name': info['name'],
                        'global_id': info['global_id'],
                        'apartment_type': info.get('apartment_type', ''),
                        'volume': info['volume'],
                        'area': info['area'],
                        'height': info['height'],
                        'rt60': info['acoustic_properties'].get('RT60'),
                        'spl': info['acoustic_properties'].get('SPL'),
                        'acoustic_risk': info.get('acoustic_risk', '')
                    })
            return summary
        except Exception as e:
            return f"Error extracting IFC spaces summary: {e}"

    def get_ifc_space_details(self, space_name_or_id):
        """Return a detailed summary for a specific space by name or GlobalId."""
        try:
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                return "No IFC file loaded."
            ifc_file = ifcopenshell.open(self.viewer.current_ifc_path)
            spaces = ifc_file.by_type("IfcSpace")
            for space in spaces:
                name = getattr(space, 'Name', '') or ''
                gid = getattr(space, 'GlobalId', '')
                if space_name_or_id == name or space_name_or_id == gid:
                    info = self.analyze_single_space(space, ifc_file)
                    return info
            return f"Space '{space_name_or_id}' not found."
        except Exception as e:
            return f"Error extracting space details: {e}"

    def format_spaces_summary_for_llm(self, summary):
        """Format the spaces summary for LLM context."""
        if isinstance(summary, str):
            return summary
        lines = ["Spaces in IFC model:"]
        for s in summary:
            lines.append(f"- {s['name']} (ID: {s['global_id']}), Type: {s['apartment_type']}, Vol: {s['volume']} m³, Area: {s['area']} m², H: {s['height']} m, RT60: {s['rt60']}, SPL: {s['spl']}, Risk: {s['acoustic_risk']}")
        return "\n".join(lines)

    def format_space_details_for_llm(self, info):
        """Format a single space's details for LLM context."""
        if not info or isinstance(info, str):
            return info
        lines = [f"Space: {info['name']} (ID: {info['global_id']})"]
        lines.append(f"Type: {info.get('apartment_type', '')}, Vol: {info['volume']} m³, Area: {info['area']} m², H: {info['height']} m")
        lines.append(f"Acoustic Properties: {info['acoustic_properties']}")
        lines.append(f"Surrounding Elements: {info['surrounding_elements']}")
        lines.append(f"Acoustic Risk: {info['acoustic_risk']}")
        return "\n".join(lines)

    def parse_ifc_to_json(self):
        """Parse the loaded IFC model to a JSON-like structure with comprehensive data extraction."""
        try:
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                print("[DEBUG] No IFC file path available")
                return None
            
            print(f"[DEBUG] Parsing IFC file: {self.viewer.current_ifc_path}")
            ifc_file = ifcopenshell.open(self.viewer.current_ifc_path)
            elements = []
            
            # Step 1: Build a mapping of space_id -> list of contained elements
            print("[DEBUG] Step 1: Building space-to-elements mapping...")
            space_to_elements = {}  # Maps space_global_id to list of element_global_ids
            element_to_space = {}   # Maps element_global_id to space_global_id
            
            # Find all IfcRelContainedInSpatialStructure relationships
            for rel in ifc_file.by_type('IfcRelContainedInSpatialStructure'):
                space_id = rel.RelatingStructure.GlobalId
                if space_id not in space_to_elements:
                    space_to_elements[space_id] = []
                
                for elem in rel.RelatedElements:
                    elem_id = elem.GlobalId
                    space_to_elements[space_id].append(elem_id)
                    element_to_space[elem_id] = space_id
                    print(f"[DEBUG] Element {elem_id} ({elem.is_a()}) belongs to space {space_id}")
            
            print(f"[DEBUG] Found {len(space_to_elements)} spaces with contained elements")
            
            # Step 2: Extract apartment information from child elements
            print("[DEBUG] Step 2: Extracting apartment information from child elements...")
            element_apartment_info = {}  # Maps element_global_id to apartment info
            
            # Get all structural elements that might contain apartment info
            wall_elements = ifc_file.by_type('IfcWall')
            slab_elements = ifc_file.by_type('IfcSlab')
            door_elements = ifc_file.by_type('IfcDoor')
            window_elements = ifc_file.by_type('IfcWindow')
            
            all_child_elements = wall_elements + slab_elements + door_elements + window_elements
            print(f"[DEBUG] Analyzing {len(all_child_elements)} child elements for apartment info")
            
            for elem in all_child_elements:
                try:
                    elem_name = getattr(elem, 'Name', None)
                    elem_global_id = getattr(elem, 'GlobalId', None)
                    elem_type = elem.is_a()
                    
                    if elem_name and elem_global_id:
                        # Extract apartment info from element name
                        apt_info = self.extract_apartment_from_element_name(elem_name)
                        if apt_info:
                            element_apartment_info[elem_global_id] = apt_info
                            print(f"[DEBUG] Found apartment info in {elem_type}: '{elem_name}' -> {apt_info}")
                            
                except Exception as elem_error:
                    print(f"[DEBUG] Error processing child element: {elem_error}")
                    continue
            
            print(f"[DEBUG] Extracted apartment info from {len(element_apartment_info)} child elements")
            
            # Step 3: Map apartment information to spaces
            print("[DEBUG] Step 3: Mapping apartment information to spaces...")
            space_apartment_mapping = {}  # Maps space_global_id to apartment info
            
            for space_id, element_ids in space_to_elements.items():
                # Find apartment info from contained elements
                space_apartment_info = None
                for elem_id in element_ids:
                    if elem_id in element_apartment_info:
                        space_apartment_info = element_apartment_info[elem_id]
                        print(f"[DEBUG] Space {space_id} gets apartment info from element {elem_id}: {space_apartment_info}")
                        break
                
                if space_apartment_info:
                    space_apartment_mapping[space_id] = space_apartment_info
                else:
                    print(f"[DEBUG] No apartment info found for space {space_id}")
            
            print(f"[DEBUG] Mapped apartment info to {len(space_apartment_mapping)} spaces")
            
            # Step 4: Process spaces with apartment information
            print("[DEBUG] Step 4: Processing spaces with apartment information...")
            spaces = ifc_file.by_type('IfcSpace')
            print(f"[DEBUG] Found {len(spaces)} IfcSpace elements")
            
            for i, space in enumerate(spaces):
                try:
                    # Extract basic space info
                    space_name = getattr(space, 'Name', None)
                    space_global_id = getattr(space, 'GlobalId', None)
                    space_long_name = getattr(space, 'LongName', None)
                    space_description = getattr(space, 'Description', None)
                    space_object_type = getattr(space, 'ObjectType', None)
                    
                    # Try multiple sources for space name
                    final_space_name = space_name or space_long_name or space_object_type or f"Space_{i+1}"
                    
                    print(f"[DEBUG] Processing space {i+1}: '{final_space_name}' (ID: {space_global_id})")
                    
                    # Get apartment info for this space
                    space_apartment_info = space_apartment_mapping.get(space_global_id)
                    
                    if space_apartment_info:
                        print(f"[DEBUG] Space '{final_space_name}' has apartment info: {space_apartment_info}")
                    else:
                        print(f"[DEBUG] No apartment info found for space '{final_space_name}'")
                    
                    space_data = {
                        'type': 'IfcSpace',
                        'name': final_space_name,
                        'global_id': space_global_id or '',
                        'long_name': space_long_name or '',
                        'description': space_description or '',
                        'object_type': space_object_type or '',
                        'properties': {},
                        'geometry': {},
                        'hardcoded': {},
                        'apartment_type': space_apartment_info.get('type', 'Unknown') if space_apartment_info else 'Unknown',
                        'apartment_info': space_apartment_info or {},
                        'contained_elements': space_to_elements.get(space_global_id, [])
                    }
                    
                    # Extract properties from the space
                    try:
                        if hasattr(space, 'IsDefinedBy'):
                            for rel in space.IsDefinedBy:
                                if rel.is_a('IfcRelDefinesByProperties'):
                                    props = rel.RelatingPropertyDefinition
                                    if props.is_a('IfcPropertySet'):
                                        for prop in props.HasProperties:
                                            if prop.is_a('IfcPropertySingleValue'):
                                                pname = prop.Name or ''
                                                val = getattr(getattr(prop, 'NominalValue', None), 'wrappedValue', None)
                                                space_data['properties'][pname] = val
                    except Exception as prop_error:
                        print(f"[DEBUG] Error extracting properties for {final_space_name}: {prop_error}")
                    
                    # Extract geometry information
                    try:
                        if hasattr(space, 'Representation') and space.Representation:
                            shape = ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(), space)
                            if shape and hasattr(shape, 'geometry'):
                                bbox = shape.geometry.BoundingBox()
                                if bbox:
                                    space_data['geometry'] = {
                                        'xmin': bbox.Xmin(), 'xmax': bbox.Xmax(),
                                        'ymin': bbox.Ymin(), 'ymax': bbox.Ymax(),
                                        'zmin': bbox.Zmin(), 'zmax': bbox.Zmax(),
                                        'width': bbox.Xmax() - bbox.Xmin(),
                                        'length': bbox.Ymax() - bbox.Ymin(),
                                        'height': bbox.Zmax() - bbox.Zmin(),
                                    }
                    except Exception as geom_error:
                        space_data['geometry'] = {'error': str(geom_error)}
                    
                    # Apply hardcoded data based on detected apartment type
                    apt_type = space_data['apartment_type']
                    if apt_type and apt_type != 'Unknown':
                        # Apply hardcoded values based on apartment type
                        hardcoded_data = {
                            '1B': {'volume': 58, 'area': 19.33, 'height': 3.0},
                            '2B': {'volume': 81, 'area': 27.00, 'height': 3.0},
                            '3B': {'volume': 108, 'area': 36.00, 'height': 3.0},
                        }
                        
                        space_data['hardcoded'] = hardcoded_data.get(apt_type, {})
                        
                        # Also add hardcoded values to properties for easy access
                        space_data['properties']['Volume'] = hardcoded_data[apt_type]['volume']
                        space_data['properties']['Area'] = hardcoded_data[apt_type]['area']
                        space_data['properties']['Height'] = hardcoded_data[apt_type]['height']
                        
                        print(f"[DEBUG] Applied hardcoded data for {apt_type}: Vol={space_data['hardcoded']['volume']}, Area={space_data['hardcoded']['area']}, H={space_data['hardcoded']['height']}")
                    else:
                        # Fallback for unknown apartment types
                        space_data['hardcoded'] = {'volume': 81, 'area': 27.00, 'height': 3.0}
                        space_data['properties']['Volume'] = 81
                        space_data['properties']['Area'] = 27.00
                        space_data['properties']['Height'] = 3.0
                        print(f"[DEBUG] Applied fallback data for unknown apartment type")
                    
                    elements.append(space_data)
                    
                except Exception as space_error:
                    print(f"[DEBUG] Error processing space {i}: {space_error}")
                    continue
            
            # Step 5: Add child elements to the result
            for elem in all_child_elements:
                try:
                    elem_name = getattr(elem, 'Name', None)
                    elem_global_id = getattr(elem, 'GlobalId', None)
                    elem_type = elem.is_a()
                    
                    child_data = {
                        'type': elem_type,
                        'name': elem_name or '',
                        'global_id': elem_global_id or '',
                        'apartment_info': element_apartment_info.get(elem_global_id),
                        'parent_space': element_to_space.get(elem_global_id)
                    }
                    elements.append(child_data)
                    
                except Exception as elem_error:
                    print(f"[DEBUG] Error processing child element: {elem_error}")
                    continue
            
            print(f"[DEBUG] Final result: {len(elements)} total elements processed")
            
            # Step 6: Show summary of apartment distribution
            apartment_counts = {}
            for space_data in elements:
                if space_data.get('type') == 'IfcSpace':
                    apt_type = space_data.get('apartment_type', 'Unknown')
                    apartment_counts[apt_type] = apartment_counts.get(apt_type, 0) + 1
            
            print(f"[DEBUG] Apartment type distribution: {apartment_counts}")
            
            return elements
            
        except Exception as e:
            print(f"[DEBUG] Error in parse_ifc_to_json: {e}")
            return {'error': str(e)}

    def apply_colors_with_clean_data(self, element_to_apartment, space_colors):
        """Apply apartment colors to the viewer shapes with acoustic failures using clean data"""
        try:
            colored_count = 0
            
            # Create a mapping from space global_id to failing status
            failing_spaces = {space_id for space_id, color in space_colors.items() if color == (1.0, 0.0, 0.0)}
            print(f"RED Found {len(failing_spaces)} failing spaces: {failing_spaces}")
            
            # If no failing spaces, just apply apartment colors
            if not failing_spaces:
                print("OK No failing spaces found - applying apartment colors only")
                for ais_shape, ifc_element in self.viewer.shape_to_ifc.items():
                    if hasattr(ifc_element, "GlobalId"):
                        global_id = ifc_element.GlobalId
                        
                        if global_id in element_to_apartment:
                            apartment_data = element_to_apartment[global_id]
                            apartment_color = apartment_data["color"]
                            
                            # Apply apartment color
                            color_obj = Quantity_Color(apartment_color[0], apartment_color[1], apartment_color[2], Quantity_TOC_RGB)
                            self.viewer._display.Context.SetColor(ais_shape, color_obj, False)
                            print(f"ART Applied {apartment_color} to element {global_id} ({apartment_data['apartment_type']})")
                            colored_count += 1
            else:
                # Apply colors with acoustic failure overlay
                for ais_shape, ifc_element in self.viewer.shape_to_ifc.items():
                    if hasattr(ifc_element, "GlobalId"):
                        global_id = ifc_element.GlobalId
                        
                        if global_id in element_to_apartment:
                            apartment_data = element_to_apartment[global_id]
                            apartment_color = apartment_data["color"]
                            space_global_id = apartment_data["space_global_id"]
                            
                            # Check if this element belongs to a failing space
                            is_failing = space_global_id in failing_spaces
                            
                            if is_failing:
                                # Apply red color for failing spaces
                                color_obj = Quantity_Color(1.0, 0.0, 0.0, Quantity_TOC_RGB)
                                self.viewer._display.Context.SetColor(ais_shape, color_obj, False)
                                print(f"RED Applied RED to element {global_id} (failing space {space_global_id})")
                            else:
                                # Apply apartment color for passing spaces
                                color_obj = Quantity_Color(apartment_color[0], apartment_color[1], apartment_color[2], Quantity_TOC_RGB)
                                self.viewer._display.Context.SetColor(ais_shape, color_obj, False)
                                print(f"ART Applied {apartment_color} to element {global_id} ({apartment_data['apartment_type']})")
                            
                            colored_count += 1
            
            # Update the display
            self.viewer._display.Context.UpdateCurrentViewer()
            self.viewer._display.Repaint()
            
            print(f"OK Applied colors to {colored_count} elements")
            
        except Exception as e:
            print(f"X Error applying colors: {e}")
            import traceback
            traceback.print_exc()

    def show_combined_acoustic_legend(self, failing_spaces, total_spaces):
        """Show legend for combined acoustic analysis overlay"""
        try:
            passing_count = total_spaces - len(failing_spaces)
            
            legend_html = f"""
            <div style="background: #2C2C2C; border: 2px solid #4CAF50; border-radius: 8px; padding: 12px; margin: 8px;">
                <h3 style="color: #4CAF50; margin: 0 0 8px 0;">MUSIC Combined Acoustic Analysis Legend</h3>
                
                <div style="display: flex; align-items: center; margin: 4px 0;">
                    <div style="width: 20px; height: 20px; background: #00FF00; border-radius: 4px; margin-right: 8px;"></div>
                    <span style="color: #FFFFFF;">GREEN Passing Spaces: {passing_count}</span>
                </div>
                
                <div style="display: flex; align-items: center; margin: 4px 0;">
                    <div style="width: 20px; height: 20px; background: #FF0000; border-radius: 4px; margin-right: 8px;"></div>
                    <span style="color: #FFFFFF;">RED Failing Spaces: {len(failing_spaces)}</span>
                </div>
                
                <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #4CAF50;">
                    <span style="color: #FFD700; font-size: 13px;">
                        IDEA <b>Legend:</b> Green = Pass, Red = Fail<br>
                        CHART <b>Total Spaces Analyzed:</b> {total_spaces}
                    </span>
                </div>
            </div>
            """
            
            self.chat_history.append(legend_html)
            
        except Exception as e:
            print(f"[DEBUG] Error showing combined legend: {e}")

    def get_apartment_color(self, apartment_type):
        """Get color for apartment type"""
        color_map = {
            "1B": (0.2, 0.6, 1.0),    # Blue
            "2B": (0.2, 0.8, 0.2),    # Green  
            "3B": (1.0, 0.4, 0.2),    # Orange
            # Also support alternative formats
            "1bed": (0.2, 0.6, 1.0),  # Blue
            "2bed": (0.2, 0.8, 0.2),  # Green
            "3bed": (1.0, 0.4, 0.2),  # Orange
            "1 bedroom": (0.2, 0.6, 1.0),  # Blue
            "2 bedroom": (0.2, 0.8, 0.2),  # Green
            "3 bedroom": (1.0, 0.4, 0.2),  # Orange
        }
        return color_map.get(apartment_type, (0.7, 0.7, 0.7))  # Default gray

    def load_json_ifc_data(self):
        """Load IFC data from JSON files in models/parse folder"""
        try:
            import json
            import os
            
            # Look for JSON files in models/parse folder
            parse_folder = "models/parse"
            json_files = []
            
            if os.path.exists(parse_folder):
                for file in os.listdir(parse_folder):
                    if file.endswith('.json'):
                        json_files.append(os.path.join(parse_folder, file))
            
            if not json_files:
                print("X No JSON files found in models/parse folder")
                return None
            
            # Load the first JSON file (you can modify this to load specific files)
            json_file = json_files[0]
            print(f"📄 Loading IFC data from: {json_file}")
            
            with open(json_file, 'r') as f:
                ifc_data = json.load(f)
            
            print(f"OK Loaded {len(ifc_data)} elements from JSON")
            return ifc_data
            
        except Exception as e:
            print(f"X Error loading JSON IFC data: {e}")
            return None

    def get_space_data_from_json(self, space_global_id, ifc_data):
        """Get space data from JSON for a specific space"""
        try:
            for element in ifc_data:
                if element.get("GlobalId") == space_global_id and element.get("IfcEntity") == "IfcSpace":
                    return {
                        "name": element.get("Name", ""),
                        "description": element.get("Description", ""),
                        "object_type": element.get("ObjectType", ""),
                        "gross_planned_area": element.get("Property_GrossPlannedArea", 0),
                        "is_external": element.get("Property_IsExternal", False),
                        "publicly_accessible": element.get("Property_PubliclyAccessible", False),
                        "handicap_accessible": element.get("Property_HandicapAccessible", False),
                        "location_x": element.get("Property_LocationX", 0),
                        "location_y": element.get("Property_LocationY", 0),
                        "location_z": element.get("Property_LocationZ", 0)
                    }
            return None
        except Exception as e:
            print(f"X Error getting space data from JSON: {e}")
            return None

    def extract_clean_ifc_data(self):
        """Extract clean IFC data with proper element-to-space mapping"""
        try:
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                print("X No IFC file loaded")
                return None
            
            # Load IFC file
            ifc_file = ifcopenshell.open(self.viewer.current_ifc_path)
            
            # Get all spaces
            spaces = ifc_file.by_type("IfcSpace")
            print(f"BUILDING Found {len(spaces)} spaces in IFC file")
            
            # Get all products (elements)
            all_products = ifc_file.by_type("IfcProduct")
            print(f"PACKAGE Found {len(all_products)} total products in IFC file")
            
            # Create clean data structure
            clean_data = {
                "spaces": {},
                "elements_by_space": {},
                "apartment_info": {},
                "statistics": {
                    "total_spaces": len(spaces),
                    "total_elements": len(all_products),
                    "mapped_elements": 0
                }
            }
            
            # Process each space
            for space in spaces:
                space_global_id = space.GlobalId
                space_name = getattr(space, "Name", "")
                long_name = getattr(space, "LongName", "")
                
                print(f"SEARCH Processing space: {space_name} ({space_global_id})")
                
                # Detect apartment type
                apartment_info = parse_apartment_info(space_name)
                if not apartment_info:
                    apartment_info = parse_apartment_info(long_name)
                
                # Store space info
                clean_data["spaces"][space_global_id] = {
                    "global_id": space_global_id,
                    "name": space_name,
                    "long_name": long_name,
                    "apartment_info": apartment_info,
                    "apartment_type": apartment_info["type"] if apartment_info else "Unknown",
                    "apartment_level": apartment_info["level"] if apartment_info else 0,
                    "apartment_number": apartment_info["number"] if apartment_info else 0,
                    "color": get_apartment_color(apartment_info["type"]) if apartment_info else (0.7, 0.7, 0.7)
                }
                
                # Store apartment info for easy access
                if apartment_info:
                    clean_data["apartment_info"][space_global_id] = apartment_info
                
                # Find all elements that belong to this space
                space_elements = self.find_elements_in_space_clean(ifc_file, space_global_id)
                
                # Store elements for this space
                clean_data["elements_by_space"][space_global_id] = []
                
                for element in space_elements:
                    element_data = {
                        "global_id": element.GlobalId,
                        "type": element.is_a(),
                        "name": getattr(element, "Name", ""),
                        "description": getattr(element, "Description", ""),
                        "object_type": getattr(element, "ObjectType", ""),
                        "space_global_id": space_global_id,
                        "apartment_type": apartment_info["type"] if apartment_info else "Unknown",
                        "apartment_level": apartment_info["level"] if apartment_info else 0,
                        "apartment_number": apartment_info["number"] if apartment_info else 0
                    }
                    
                    # Add hardcoded apartment data
                    if apartment_info and apartment_info["type"] == "1B":
                        element_data.update({
                            "hardcoded_volume": 45,
                            "hardcoded_area": 45,
                            "hardcoded_height": 3.0
                        })
                    elif apartment_info and apartment_info["type"] == "2B":
                        element_data.update({
                            "hardcoded_volume": 60,
                            "hardcoded_area": 60,
                            "hardcoded_height": 3.0
                        })
                    elif apartment_info and apartment_info["type"] == "3B":
                        element_data.update({
                            "hardcoded_volume": 75,
                            "hardcoded_area": 75,
                            "hardcoded_height": 3.0
                        })
                    else:
                        element_data.update({
                            "hardcoded_volume": 50,
                            "hardcoded_area": 50,
                            "hardcoded_height": 3.0
                        })
                    
                    clean_data["elements_by_space"][space_global_id].append(element_data)
                
                print(f"   PACKAGE Found {len(space_elements)} elements in space {space_name}")
                clean_data["statistics"]["mapped_elements"] += len(space_elements)
            
            print(f"OK Clean data extraction complete:")
            print(f"   BUILDING Spaces: {clean_data['statistics']['total_spaces']}")
            print(f"   PACKAGE Total elements: {clean_data['statistics']['total_elements']}")
            print(f"   🎯 Mapped elements: {clean_data['statistics']['mapped_elements']}")
            
            return clean_data
            
        except Exception as e:
            print(f"X Error extracting clean IFC data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def find_elements_in_space_clean(self, ifc_file, space_global_id):
        """Find all elements that belong to a specific space using proper IFC relationships"""
        elements = []
        
        try:
            # Get the space object
            space = ifc_file.by_guid(space_global_id)
            
            # Method 1: Check IfcRelContainedInSpatialStructure
            if hasattr(space, "ContainsElements"):
                for rel in space.ContainsElements:
                    for element in rel.RelatedElements:
                        if element not in elements:
                            elements.append(element)
            
            # Method 2: Check IfcRelSpaceBoundary
            if hasattr(space, "BoundedBy"):
                for rel in space.BoundedBy:
                    if hasattr(rel, "RelatedBuildingElement"):
                        element = rel.RelatedBuildingElement
                        if element not in elements:
                            elements.append(element)
            
            # Method 3: Check by spatial structure hierarchy (traverse up)
            all_products = ifc_file.by_type("IfcProduct")
            for product in all_products:
                if hasattr(product, "ContainedInStructure"):
                    for rel in product.ContainedInStructure:
                        if rel.RelatingStructure.GlobalId == space_global_id:
                            if product not in elements:
                                elements.append(product)
            
            # Method 4: Check by spatial decomposition (traverse down)
            if hasattr(space, "IsDecomposedBy"):
                for rel in space.IsDecomposedBy:
                    for sub_space in rel.RelatedObjects:
                        if hasattr(sub_space, "ContainsElements"):
                            for sub_rel in sub_space.ContainsElements:
                                for element in sub_rel.RelatedElements:
                                    if element not in elements:
                                        elements.append(element)
            
        except Exception as e:
            print(f"WARNING Error finding elements for space {space_global_id}: {e}")
        
        return elements

    def get_ifc_context_for_llm(self):
        """Get IFC context for LLM using clean data extraction"""
        try:
            # Use the new clean data extraction
            clean_data = self.extract_clean_ifc_data()
            if not clean_data:
                print("WARNING Could not extract clean IFC data, using fallback")
                return self.get_fallback_ifc_context()
            
            # Build context from clean data
            context = {
                "model_info": {
                    "total_spaces": clean_data["statistics"]["total_spaces"],
                    "total_elements": clean_data["statistics"]["total_elements"],
                    "mapped_elements": clean_data["statistics"]["mapped_elements"],
                    "file_path": self.viewer.current_ifc_path if hasattr(self, 'viewer') else "Unknown"
                },
                "spaces": {},
                "apartments": {},
                "elements": {}
            }
            
            # Process spaces and their elements
            for space_global_id, space_data in clean_data["spaces"].items():
                space_name = space_data["name"]
                apartment_type = space_data["apartment_type"]
                
                # Add space info
                context["spaces"][space_global_id] = {
                    "name": space_name,
                    "long_name": space_data["long_name"],
                    "apartment_type": apartment_type,
                    "apartment_level": space_data["apartment_level"],
                    "apartment_number": space_data["apartment_number"],
                    "hardcoded_volume": space_data.get("hardcoded_volume", 50),
                    "hardcoded_area": space_data.get("hardcoded_area", 50),
                    "hardcoded_height": space_data.get("hardcoded_height", 3.0)
                }
                
                # Group by apartment type
                if apartment_type not in context["apartments"]:
                    context["apartments"][apartment_type] = {
                        "count": 0,
                        "spaces": [],
                        "total_elements": 0
                    }
                
                context["apartments"][apartment_type]["count"] += 1
                context["apartments"][apartment_type]["spaces"].append(space_name)
                
                # Add elements for this space
                space_elements = clean_data["elements_by_space"].get(space_global_id, [])
                context["apartments"][apartment_type]["total_elements"] += len(space_elements)
                
                for element_data in space_elements:
                    element_global_id = element_data["global_id"]
                    context["elements"][element_global_id] = {
                        "type": element_data["type"],
                        "name": element_data["name"],
                        "description": element_data["description"],
                        "object_type": element_data["object_type"],
                        "space_global_id": space_global_id,
                        "space_name": space_name,
                        "apartment_type": apartment_type,
                        "apartment_level": element_data["apartment_level"],
                        "apartment_number": element_data["apartment_number"],
                        "hardcoded_volume": element_data["hardcoded_volume"],
                        "hardcoded_area": element_data["hardcoded_area"],
                        "hardcoded_height": element_data["hardcoded_height"]
                    }
            
            print(f"OK Built IFC context from clean data:")
            print(f"   BUILDING Spaces: {len(context['spaces'])}")
            print(f"   🏠 Apartment types: {list(context['apartments'].keys())}")
            print(f"   PACKAGE Elements: {len(context['elements'])}")
            
            return context
            
        except Exception as e:
            print(f"X Error building IFC context from clean data: {e}")
            import traceback
            traceback.print_exc()
            return self.get_fallback_ifc_context()

    def load_preloaded_models(self):
        """Load preloaded models from the models/ folder"""
        try:
            models_dir = "models"
            if not os.path.exists(models_dir):
                print(f"WARNING Models directory '{models_dir}' not found")
                return
            
            # Clear existing items except the first one
            while self.preloaded_combo.count() > 1:
                self.preloaded_combo.removeItem(1)
            
            # Find all IFC files in the models directory
            ifc_files = []
            for file in os.listdir(models_dir):
                if file.lower().endswith('.ifc'):
                    ifc_files.append(file)
            
            ifc_files.sort()  # Sort alphabetically
            
            if not ifc_files:
                print(f"WARNING No IFC files found in '{models_dir}' directory")
                return
            
            # Add IFC files to combo box
            for ifc_file in ifc_files:
                self.preloaded_combo.addItem(ifc_file)
            
            print(f"OK Loaded {len(ifc_files)} preloaded models: {ifc_files}")
            
        except Exception as e:
            print(f"X Error loading preloaded models: {e}")
            import traceback
            traceback.print_exc()

    def on_preloaded_model_selected(self, model_name):
        """Handle preloaded model selection"""
        try:
            if model_name == "Select a preloaded model..." or not model_name:
                return
            
            model_path = os.path.join("models", model_name)
            if not os.path.exists(model_path):
                print(f"X Model file not found: {model_path}")
                return
            
            print(f"🔄 Loading preloaded model: {model_name}")
            
            # Load the model using the existing upload handler
            self.viewer.load_ifc_file(model_path)
            
            # Update the refresh button state
            if hasattr(self, 'refresh_btn'):
                self.refresh_btn.setEnabled(True)
            
            print(f"OK Successfully loaded preloaded model: {model_name}")
            
        except Exception as e:
            print(f"X Error loading preloaded model: {e}")
            import traceback
            traceback.print_exc()

    def get_detailed_acoustic_suggestions(self, space_info, failures):
        """Get detailed acoustic improvement suggestions with specific materials and solutions"""
        try:
            # Get base recommendations
            base_recommendations = self.get_acoustic_recommendations(space_info, failures)
            
            # Add specific material database
            material_database = {
                "wall_panels": {
                    "fiberglass": {"absorption": 0.85, "cost": "$15-25/sqft", "thickness": "2-4 inches"},
                    "mineral_wool": {"absorption": 0.80, "cost": "$12-20/sqft", "thickness": "2-3 inches"},
                    "fabric_wrapped": {"absorption": 0.75, "cost": "$18-30/sqft", "thickness": "1-2 inches"},
                    "perforated_wood": {"absorption": 0.60, "cost": "$25-40/sqft", "thickness": "0.5-1 inch"}
                },
                "ceiling_tiles": {
                    "mineral_fiber": {"absorption": 0.80, "cost": "$8-15/sqft", "thickness": "0.5-1 inch"},
                    "fiberglass": {"absorption": 0.85, "cost": "$10-18/sqft", "thickness": "1-2 inches"},
                    "perforated_metal": {"absorption": 0.70, "cost": "$12-22/sqft", "thickness": "0.25-0.5 inch"}
                },
                "flooring": {
                    "carpet_medium": {"absorption": 0.40, "cost": "$6-12/sqft", "thickness": "0.25-0.5 inch"},
                    "carpet_heavy": {"absorption": 0.60, "cost": "$8-15/sqft", "thickness": "0.5-0.75 inch"},
                    "acoustic_underlayment": {"absorption": 0.30, "cost": "$3-8/sqft", "thickness": "0.25-0.5 inch"}
                },
                "windows": {
                    "double_glazed": {"stc": 35, "cost": "$45-65/sqft", "thickness": "1-1.5 inches"},
                    "laminated": {"stc": 40, "cost": "$55-75/sqft", "thickness": "0.5-1 inch"},
                    "acoustic_sealant": {"stc_improvement": 5, "cost": "$2-5/linear ft", "application": "Perimeter sealing"}
                }
            }
            
            # Generate specific suggestions based on failures
            specific_suggestions = []
            
            for failure in failures:
                if 'LAeq' in failure:
                    specific_suggestions.append(f"""
**NOISE LEVEL REDUCTION SUGGESTIONS:**
• **Immediate (Week 1):** Install {material_database['wall_panels']['fiberglass']['absorption']} absorption fiberglass panels
  - Cost: {material_database['wall_panels']['fiberglass']['cost']}
  - Expected improvement: 8-12 dB reduction
  - Installation: 2-3 days

• **Short-term (Week 2):** Upgrade windows to {material_database['windows']['laminated']['stc']} STC laminated glass
  - Cost: {material_database['windows']['laminated']['cost']}
  - Expected improvement: 15-20 dB reduction
  - Installation: 1-2 days per window

• **Medium-term (Week 3):** Apply {material_database['windows']['acoustic_sealant']['stc_improvement']} STC improvement acoustic sealant
  - Cost: {material_database['windows']['acoustic_sealant']['cost']}
  - Expected improvement: 5 dB additional reduction
  - Installation: 2-4 hours per window
""")
                
                if 'RT60' in failure:
                    specific_suggestions.append(f"""
**REVERBERATION CONTROL SUGGESTIONS:**
• **Ceiling Treatment:** Install {material_database['ceiling_tiles']['mineral_fiber']['absorption']} absorption mineral fiber tiles
  - Cost: {material_database['ceiling_tiles']['mineral_fiber']['cost']}
  - Expected improvement: 0.3-0.5s RT60 reduction
  - Installation: 3-5 days

• **Wall Treatment:** Apply {material_database['wall_panels']['fabric_wrapped']['absorption']} absorption fabric panels to 30-40% of walls
  - Cost: {material_database['wall_panels']['fabric_wrapped']['cost']}
  - Expected improvement: 0.2-0.4s RT60 reduction
  - Installation: 2-3 days

• **Floor Treatment:** Install {material_database['flooring']['carpet_heavy']['absorption']} absorption heavy carpet
  - Cost: {material_database['flooring']['carpet_heavy']['cost']}
  - Expected improvement: 0.1-0.2s RT60 reduction
  - Installation: 2-3 days
""")
            
            # Add cost-benefit analysis
            cost_benefit = f"""
**COST-BENEFIT ANALYSIS:**
• **Total Investment Range:** $2,000 - $8,000 for typical apartment
• **Expected Acoustic Improvement:** 15-25 dB noise reduction, 0.4-0.8s RT60 improvement
• **ROI Timeline:** 6-12 months through improved tenant satisfaction and reduced complaints
• **Maintenance:** Annual inspection and cleaning of acoustic treatments
• **Lifespan:** 10-15 years for most treatments

**IMPLEMENTATION STRATEGY:**
1. **Phase 1 (Week 1-2):** Critical noise reduction - $1,500-3,000
2. **Phase 2 (Week 3-4):** Reverberation control - $1,000-2,500
3. **Phase 3 (Week 5-6):** Finishing touches - $500-1,500

**SUPPLIER RECOMMENDATIONS:**
• Acoustic panels: Armstrong, USG, or local acoustic suppliers
• Window treatments: Local glazing contractors with acoustic expertise
• Installation: Certified acoustic contractors or experienced general contractors
"""
            
            return base_recommendations + "\n" + "\n".join(specific_suggestions) + cost_benefit
            
        except Exception as e:
            return f"Error generating detailed suggestions: {e}"

    def force_refresh_ifc_data(self):
        """Force refresh the IFC data parsing."""
        try:
            if hasattr(self.viewer, 'current_ifc_path') and self.viewer.current_ifc_path:
                print(f"\n[DEBUG] Force refreshing IFC data from: {self.viewer.current_ifc_path}")
                self.chat_history.append(f"🔄 Refreshing IFC data from: {os.path.basename(self.viewer.current_ifc_path)}")
                
                # Re-parse IFC data
                self.ifc_json_data = self.parse_ifc_to_json()
                
                if self.ifc_json_data and isinstance(self.ifc_json_data, list):
                    spaces = [e for e in self.ifc_json_data if e.get('type') == 'IfcSpace']
                    apt_types = {}
                    for space in spaces:
                        apt_type = space.get('apartment_type', 'Unknown')
                        apt_types[apt_type] = apt_types.get(apt_type, 0) + 1
                    
                    print(f"[DEBUG] Refresh results: {len(spaces)} spaces, apartment types: {apt_types}")
                    
                    # Show refresh results in chat
                    refresh_summary = f"OK Refresh Complete:\n"
                    refresh_summary += f"• Total elements: {len(self.ifc_json_data)}\n"
                    refresh_summary += f"• Spaces found: {len(spaces)}\n"
                    refresh_summary += f"• Apartment types: {apt_types}\n"
                    self.chat_history.append(refresh_summary)
                    
                    # Show sample spaces with apartment types
                    if spaces:
                        sample_spaces = []
                        for space in spaces[:5]:
                            name = space.get('name', 'Unknown')
                            apt_type = space.get('apartment_type', 'Unknown')
                            volume = space.get('hardcoded', {}).get('volume', 'N/A')
                            area = space.get('hardcoded', {}).get('area', 'N/A')
                            sample_spaces.append(f"• {name} -> {apt_type} (Vol: {volume}m³, Area: {area}m²)")
                        
                        if sample_spaces:
                            sample_text = f"📋 Sample Spaces:\n" + "\n".join(sample_spaces)
                            self.chat_history.append(sample_text)
                    
                    return True
                else:
                    print(f"[DEBUG] Refresh failed: {self.ifc_json_data}")
                    self.chat_history.append(f"X Refresh failed: {self.ifc_json_data}")
                    return False
            else:
                print(f"[DEBUG] No IFC file loaded for refresh")
                self.chat_history.append(f"X No IFC file loaded for refresh")
                return False
        except Exception as e:
            error_msg = f"X Error during refresh: {str(e)}"
            print(f"[DEBUG] {error_msg}")
            self.chat_history.append(error_msg)
            return False

    def show_acoustic_analysis_with_overlay(self):
        """Combined function: Show apartment overlay with acoustic failure highlighting"""
        try:
            # Check if IFC file is loaded
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                self.chat_history.append("<span style='color:red;'>X No IFC file loaded. Please load an IFC file first.</span>")
                return
            
            # Show analysis starting message
            self.chat_history.append("<span style='color:#FF9500;'><b>SEARCH Starting Combined Acoustic Analysis...</b></span>")
            
            # Determine if analyze_all is checked
            analyze_all = False
            if hasattr(self, 'analyze_all_checkbox'):
                analyze_all = self.analyze_all_checkbox.isChecked()
            
            # Run the analysis with no timeout
            import threading
            result = None
            analysis_complete = threading.Event()
            
            def run_analysis():
                nonlocal result
                try:
                    result = self.identify_failing_acoustic_spaces(analyze_all=analyze_all)
                    analysis_complete.set()
                except Exception as e:
                    result = f"Analysis error: {str(e)}"
                    analysis_complete.set()
            
            analysis_thread = threading.Thread(target=run_analysis)
            analysis_thread.daemon = True
            analysis_thread.start()
            
            # Wait for analysis to complete (no timeout)
            analysis_complete.wait()
            
            if result:
                # Display results
                self.chat_history.append(f"<span style='color:#FF9500;'><b>SEARCH Acoustic Analysis Results:</b></span>")
                self.chat_history.append(f"<span style='color:#FDF6F6;'>{result}</span>")
                
                # Create combined overlay: apartment types + acoustic failures
                self.create_combined_acoustic_overlay(analyze_all=analyze_all)
            else:
                self.chat_history.append("<span style='color:red;'>X Analysis failed to produce results.</span>")
                
        except Exception as e:
            self.chat_history.append(f"<span style='color:red;'>X Error in acoustic analysis: {str(e)}</span>")

    def create_combined_acoustic_overlay(self, analyze_all=False):
        """Create combined overlay showing apartment types and highlighting failing acoustic spaces"""
        try:
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                self.chat_history.append("<span style='color:red;'>X No IFC file loaded for overlay</span>")
                return
            
            self.chat_history.append("<span style='color:#4CAF50;'><b>ART Creating combined apartment + acoustic overlay...</b></span>")
            
            # Load IFC file
            ifc_file = ifcopenshell.open(self.viewer.current_ifc_path)
            spaces = ifc_file.by_type("IfcSpace")
            print(f"BUILDING Found {len(spaces)} spaces in IFC file")
            
            # Get all products for element mapping
            all_products = ifc_file.by_type("IfcProduct")
            print(f"PACKAGE Found {len(all_products)} total products in IFC file")
            
            # Create element-to-apartment mapping
            element_to_apartment = {}
            space_colors = {}
            failing_spaces = set()
            
            # Process each space
            for space in spaces:
                space_global_id = space.GlobalId
                space_name = getattr(space, "Name", "")
                long_name = getattr(space, "LongName", "")
                
                print(f"SEARCH Processing space: {space_name} ({space_global_id})")
                
                # Detect apartment type
                apartment_info = parse_apartment_info(space_name)
                if not apartment_info:
                    apartment_info = parse_apartment_info(long_name)
                
                # Get apartment type string
                apartment_type = "Unknown"
                if apartment_info:
                    apartment_type = apartment_info.get("type", "Unknown")
                
                # Get apartment color
                apartment_color = self.get_apartment_color(apartment_type)
                
                # Run acoustic analysis on this space
                space_info = self.analyze_single_space(space, ifc_file)
                if space_info:
                    # Add apartment info to space_info
                    space_info['apartment_type'] = apartment_type
                    space_info['apartment_info'] = apartment_info
                    
                    # Check for acoustic failures using UI10 logic
                    failures = self.check_acoustic_failures(space_info)
                    
                    if failures:
                        # Space has acoustic failures - mark as failing
                        failing_spaces.add(space_global_id)
                        space_colors[space_global_id] = (1.0, 0.0, 0.0)  # Red for failing
                        print(f"RED Space {space_name} has {len(failures)} acoustic failures")
                    else:
                        # Space passes acoustic checks - use apartment color
                        space_colors[space_global_id] = apartment_color
                        print(f"OK Space {space_name} passes acoustic checks")
                
                # Find elements in this space
                elements_in_space = self.find_elements_in_space_clean(ifc_file, space_global_id)
                print(f"   PACKAGE Found {len(elements_in_space)} elements in space {space_name}")
                
                # Map elements to apartment data
                for element in elements_in_space:
                    if hasattr(element, "GlobalId"):
                        element_global_id = element.GlobalId
                        element_to_apartment[element_global_id] = {
                            "space_global_id": space_global_id,
                            "space_name": space_name,
                            "apartment_type": apartment_type,
                            "apartment_info": apartment_info,
                            "color": apartment_color,
                            "acoustic_failures": failures if 'failures' in locals() else []
                        }
            
            # Apply colors to the viewer
            self.apply_colors_with_clean_data(element_to_apartment, space_colors)
            
            # Show legend
            self.show_combined_acoustic_legend(failing_spaces, len(spaces))
            
            # Show summary in chat
            passing_count = len(spaces) - len(failing_spaces)
            summary_html = f"""
            <div style="background: #2C2C2C; border: 2px solid #4CAF50; border-radius: 8px; padding: 12px; margin: 8px;">
                <h3 style="color: #4CAF50; margin: 0 0 8px 0;">MUSIC Combined Acoustic Analysis Complete</h3>
                <p style="color: #FFFFFF; margin: 4px 0;">CHART <b>Results:</b></p>
                <ul style="color: #FFFFFF; margin: 4px 0;">
                    <li>GREEN <b>Passing Spaces:</b> {passing_count}</li>
                    <li>RED <b>Failing Spaces:</b> {len(failing_spaces)}</li>
                    <li>PACKAGE <b>Total Elements Mapped:</b> {len(element_to_apartment)}</li>
                </ul>
                <p style="color: #FFD700; font-size: 13px; margin: 8px 0 0 0;">
                    IDEA <b>Legend:</b> Green = Pass, Red = Fail, Colors = Apartment Types
                </p>
            </div>
            """
            self.chat_history.append(summary_html)
            
            print(f"OK Combined overlay complete: {passing_count} passing, {len(failing_spaces)} failing spaces")
            
        except Exception as e:
            error_msg = f"X Combined overlay creation failed: {e}"
            print(error_msg)
            self.chat_history.append(f"<span style='color:red;'>{error_msg}</span>")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
        app = QApplication(sys.argv)
        win = EcoformMainWindow()
        win.show()
        sys.exit(app.exec())
