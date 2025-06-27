import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs("Reports", exist_ok=True)
import pandas as pd
import sqlite3

from server.config import client, completion_model, embedding_model


from scripts.core.acoustic_pipeline import run_pipeline, run_from_free_text
from scripts.core.llm_calls import extract_variables, build_answer

from scripts.core.recommend_recompute import recommend_recompute
from scripts.core.sql_query_handler import handle_llm_query as handle_llm_sql
from scripts.core.llm_acoustic_query_handler import handle_llm_query as handle_llm_acoustic
from scripts.core.fix_occ_import_error import InfoDock, EnhancedInfoDock

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
    QDockWidget, QTextBrowser, QGroupBox, QSlider, QCheckBox  # Added enhanced widgets
)

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
    os.add_dll_directory(dll_path)  # ✅ Only works in Python 3.8+
else:
    os.environ["PATH"] = dll_path + os.pathsep + os.environ["PATH"]

try:
    from OCC.Display.backend import load_backend
    print("✅ OCC module loaded successfully!")
except ModuleNotFoundError:
    print("❌ OCC module not found. Trying fallback import...")
    try:
        import OCC
        print("✅ OCC base module available (but submodules missing). Try checking submodule availability.")
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
        
        print("✅ Enhanced rendering settings applied")
        
    except Exception as e:
        print(f"⚠️ Could not apply all enhanced settings: {e}")

def improve_shape_tessellation(shape, quality=0.1):
    """Improve the tessellation quality of a shape"""
    try:
        # Create incremental mesh with better quality
        mesh = BRepMesh_IncrementalMesh(shape, quality, False, quality, True)
        mesh.Perform()
        
        if mesh.IsDone():
            print(f"✅ Shape quality improved with tessellation quality: {quality}")
            return shape
        else:
            print(f"⚠️ Could not improve shape quality")
            return shape
            
    except Exception as e:
        print(f"⚠️ Error improving shape quality: {e}")
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
        print(f"⚠️ Could not apply enhanced material: {e}")

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
        print(f"⚠️ Error finding elements for space {space_global_id}: {e}")
    
    return elements

def create_apartment_overlay(viewer, ifc_file_path):
    """Create apartment color overlay for all elements"""
    try:
        # Load IFC file
        ifc_file = ifcopenshell.open(ifc_file_path)
        
        # Find apartment spaces
        apartment_spaces = find_apartment_spaces(ifc_file)
        print(f"🏢 Found {len(apartment_spaces)} apartment spaces")
        
        # Create mapping of elements to apartment colors
        element_to_apartment = {}
        apartment_elements = defaultdict(list)
        
        # Process each apartment space
        for space_global_id, space_data in apartment_spaces.items():
            apt_info = space_data["info"]
            color = space_data["color"]
            
            print(f"🎨 Processing apartment {apt_info['type']} on level {apt_info['level']}")
            
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
            
            print(f"   📦 Found {len(elements)} elements in apartment {apt_info['type']}")
        
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
        print(f"❌ Error creating apartment overlay: {e}")
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
        
        print(f"✅ Applied apartment colors to {colored_count} elements")
        
    except Exception as e:
        print(f"❌ Error applying colors to viewer: {e}")

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
        print("✅ Reset to original element colors")
        
    except Exception as e:
        print(f"❌ Error resetting colors: {e}")

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

# ---- Styles ----
panel_style = f"background: {BG_PANEL}; border-radius: 12px; padding: 10px;"
btn_style = f"""
    QPushButton {{
        background: {ACCENT};
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
        color: {TEXT_ACCENT};
        border-radius: 12px;
        font-size: 16px;
        border: 1px solid {BORDER_ACCENT};
        padding: 10px;
    }}
    QLineEdit {{
        background: #24282c;
        color: {TEXT_ACCENT};
        border-radius: 8px;
        font-size: 16px;
        border: 1px solid {BORDER_ACCENT};
        padding: 8px;
    }}
"""
tab_style = f"""
    QTabWidget::pane {{
        border: none;
    }}
    QTabBar::tab {{
        background: {BG_PANEL};
        color: {TEXT_ACCENT};
        border-radius: 10px;
        font-size: 13px;
        margin: 0 6px;
        padding: 10px 20px;
    }}
    QTabBar::tab:selected {{
        background: {ACCENT};
        color: #23262B;
        font-weight: bold;
    }}
"""

class Neo4jViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.web_view = QWebEngineView()
        self.web_view.load(QUrl("http://localhost:7474/browser/"))

        self.graph_btn = QPushButton("Show Preprogrammed Graph")
        self.graph_btn.clicked.connect(self.show_custom_graph)

        layout = QVBoxLayout()
        layout.addWidget(self.web_view)
        layout.addWidget(self.graph_btn)
        self.setLayout(layout)

    def show_custom_graph(self):
        win = QWebEngineView()
        win.setWindowTitle("Custom Graph")
        win.resize(1000, 700)
        win.load(QUrl.fromLocalFile('/absolute/path/to/your/graph.html'))
        win.show()

    def show_custom_graph(self):
        # Open a new window with your HTML
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
            print("✅ Enhanced rendering enabled")
        except Exception as e:
            print(f"❌ Failed to enable enhanced rendering: {e}")

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
                        print(f"⚠️ Error processing {product.is_a()}: {str(e)[:100]}...")
                    continue
            
            # Update display
            self._display.Context.UpdateCurrentViewer()
            self._display.FitAll()
            
            print(f"✅ Loaded IFC with enhanced rendering: {loaded_count} elements (skipped {error_count} errors)")
            
            if loaded_count == 0:
                print("⚠️ No geometry loaded. This might be due to:")
                print("   - IFC file has no geometric representation")
                print("   - Elements are not in supported types")
                print("   - Display driver issues")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading IFC with enhanced rendering: {e}")
            return False

    def create_apartment_overlay(self):
        """Create apartment color overlay"""
        if not self.current_ifc_path:
            print("❌ No IFC file loaded")
            return None
            
        try:
            stats = create_apartment_overlay(self, self.current_ifc_path)
            if stats:
                self.apartment_overlay_enabled = True
                print(f"✅ Apartment overlay created: {stats}")
                return stats
            else:
                print("❌ Failed to create apartment overlay")
                return None
        except Exception as e:
            print(f"❌ Error creating apartment overlay: {e}")
            return None

    def reset_to_element_colors(self):
        """Reset colors to original element type colors"""
        try:
            reset_apartment_colors(self)
            self.apartment_overlay_enabled = False
            print("✅ Reset to element type colors")
        except Exception as e:
            print(f"❌ Error resetting colors: {e}")

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
        
        print(f"✅ Saved {len(data)} elements to {db_path}")
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
                print("❌ Display not initialized. Initializing now...")
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
            print(f"❌ Error in load_ifc_file: {e}")
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
                            print(f"⚠️ Failed to create AIS shape for {product.is_a()}: {str(shape_error)[:50]}...")
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
                            print(f"📦 Loaded {loaded_count} elements...")
                        
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:  # Only show first 5 errors to avoid spam
                        print(f"⚠️ Error processing {product.is_a()}: {str(e)[:100]}...")
                    continue
            
            # Update display
            if hasattr(self, '_display') and self._display is not None:
                self._display.Context.UpdateCurrentViewer()
                self._display.FitAll()
            
            print(f"✅ Loaded IFC: {loaded_count} elements (skipped {error_count} errors)")
            
            if loaded_count == 0:
                print("⚠️ No geometry loaded. This might be due to:")
                print("   - IFC file has no geometric representation")
                print("   - Elements are not in supported types")
                print("   - Display driver issues")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading IFC: {e}")
            return False

    def generate_heatmap(self):
        # This would generate a heatmap visualization
        # For now, just print a message
        print("🔍 Heatmap generation not implemented yet")
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
            print(f"❌ Error analyzing IFC file: {e}")
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
            print(f"⚠️ Error creating space bounding box: {e}")
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
            print(f"⚠️ Error getting space bounds: {e}")
            return None

class EcoformMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from scripts.core.neo4j_interface import Neo4jConnector
        self.neo4j = Neo4jConnector(password="123456789")  # 🔑 Replace with your actual password
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
        title.setStyleSheet(f"background: {COLOR_TITLE_BG}; color: {COLOR_TITLE_TEXT}; border-radius: 10px; padding: 18px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(title)

        # --- Tabs for Inputs ---
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; }}
            QTabBar::tab {{
                background: {COLOR_TAB_BG};
                color: {COLOR_TAB_TEXT};
                font-weight: bold; padding: 8px 20px; border-radius: 10px;
            }}
            QTabBar::tab:selected {{ background: {COLOR_DROPDOWN_FRAME}; }}
        """)

        # --- Geometry Inputs Tab ---
        geo_tab = QWidget()
        geo_layout = QVBoxLayout(geo_tab)
        geo_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        geo_layout.addWidget(QLabel("Apartment Type Group:", font=FONT_LABEL))
        self.geo_apartment = QComboBox()
        self.geo_apartment.addItems(["1Bed", "2Bed", "3Bed"])
        self.geo_apartment.setStyleSheet(f"background: {COLOR_DROPDOWN_BG}; color: {COLOR_INPUT_TEXT}; border: 2px solid {COLOR_DROPDOWN_FRAME};")
        geo_layout.addWidget(self.geo_apartment)

        geo_layout.addWidget(QLabel("Wall Material:", font=FONT_LABEL))
        self.geo_wall = QComboBox()
        self.geo_wall.addItems([
            "Painted Brick", "Unpainted Brick", "Concrete Block (Coarse)", "Concrete Block (Painted)",
            "Gypsum Board", "Plaster on Masonry", "Plaster with Wallpaper Backing", "Wood Paneling",
            "Acoustic Plaster", "Fiberglass Board"])
        self.geo_wall.setStyleSheet(f"background: {COLOR_DROPDOWN_BG}; color: {COLOR_INPUT_TEXT}; border: 2px solid {COLOR_DROPDOWN_FRAME};")
        geo_layout.addWidget(self.geo_wall)

        geo_layout.addWidget(QLabel("Window Material:", font=FONT_LABEL))
        self.geo_window = QComboBox()
        self.geo_window.addItems([
            "Single Pane Glass", "Double Pane Glass", "Laminated Glass", "Wired Glass", "Frosted Glass",
            "Insulated Glazing Unit", "Glass Block", "Glazed Ceramic Tile", "Large Pane Glass", "Small Pane Glass"])
        self.geo_window.setStyleSheet(f"background: {COLOR_DROPDOWN_BG}; color: {COLOR_INPUT_TEXT}; border: 2px solid {COLOR_DROPDOWN_FRAME};")
        geo_layout.addWidget(self.geo_window)

        geo_layout.addWidget(QLabel("Floor Material:", font=FONT_LABEL))
        self.geo_floor = QComboBox()
        self.geo_floor.addItems([
            "Marble", "Terrazzo", "Vinyl Tile", "Wood Parquet", "Wood Flooring on Joists",
            "Thin Carpet on Concrete", "Thin Carpet on Wood", "Medium Pile Carpet", "Thick Pile Carpet", "Cork Floor Tiles"])
        self.geo_floor.setStyleSheet(f"background: {COLOR_DROPDOWN_BG}; color: {COLOR_INPUT_TEXT}; border: 2px solid {COLOR_DROPDOWN_FRAME};")
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
        self.scenario_box.setStyleSheet(f"background: {COLOR_DROPDOWN_BG}; color: {COLOR_INPUT_TEXT}; border: 2px solid {COLOR_DROPDOWN_FRAME};")
        scenario_layout.addWidget(self.scenario_box)

        scenario_layout.addWidget(QLabel("Zone:", font=FONT_LABEL))
        self.zone_box = QComboBox()
        self.zone_box.addItems([
            "HD-Urban-V1", "MD-Urban-V2", "LD-Urban-V3", "Ind-Zone-V0",
            "Roadside-V1", "Roadside-V2", "Roadside-V3", "GreenEdge-V3"])
        self.zone_box.setStyleSheet(f"background: {COLOR_DROPDOWN_BG}; color: {COLOR_INPUT_TEXT}; border: 2px solid {COLOR_DROPDOWN_FRAME};")
        scenario_layout.addWidget(self.zone_box)

        scenario_layout.addWidget(QLabel("Activity:", font=FONT_LABEL))
        self.activity_box = QComboBox()
        self.activity_box.addItems([
            "Sleeping", "Working", "Living", "Dining", "Learning", "Healing", "Exercise", "Co-working"])
        self.activity_box.setStyleSheet(f"background: {COLOR_DROPDOWN_BG}; color: {COLOR_INPUT_TEXT}; border: 2px solid {COLOR_DROPDOWN_FRAME};")
        scenario_layout.addWidget(self.activity_box)

        scenario_layout.addWidget(QLabel("Time Period:", font=FONT_LABEL))
        self.time_box = QComboBox()
        self.time_box.addItems(["Day", "Night"])
        self.time_box.setStyleSheet(f"background: {COLOR_DROPDOWN_BG}; color: {COLOR_INPUT_TEXT}; border: 2px solid {COLOR_DROPDOWN_FRAME};")
        scenario_layout.addWidget(self.time_box)

        scenario_layout.addWidget(QLabel("Custom Sound Upload (WAV):", font=FONT_LABEL))
        self.upload_btn = QPushButton("Upload WAV File")
        self.upload_btn.setStyleSheet(f"background: {COLOR_INPUT_BG}; color: {COLOR_INPUT_TEXT}; font-weight: bold;")
        scenario_layout.addWidget(self.upload_btn)

        tabs.addTab(scenario_tab, "Scenario Inputs")
        left_layout.addWidget(tabs)

        self.threeD_btn = QPushButton("3D File Upload")
        self.threeD_btn.setStyleSheet(f"background: {COLOR_INPUT_BG}; color: {COLOR_INPUT_TEXT}; font-weight: bold;")
        self.threeD_btn.clicked.connect(self.handle_ifc_upload)
        left_layout.addWidget(self.threeD_btn)

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
        self.chat_history.setStyleSheet(f"background: {COLOR_CHAT_BG}; color: {COLOR_CHAT_TEXT}; font-weight: bold;")
        self.chat_layout.addWidget(self.chat_history)
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your question...")
        self.chat_layout.addWidget(self.chat_input)
        self.send_btn = QPushButton("Send")
        self.send_btn.setStyleSheet(f"background: {COLOR_INPUT_BG}; color: {COLOR_INPUT_TEXT}; font-weight: bold;")
        self.send_btn.clicked.connect(self.on_chatbot_send)
        self.chat_layout.addWidget(self.send_btn)
        # Right panel layout
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Add Neo4j Viewer as tab or embedded widget
        self.neo4j_viewer = Neo4jViewer(self)
        right_layout.addWidget(self.neo4j_viewer)
        self.neo4j.upload_from_csv("Reports/nodes.csv", "Reports/edges.csv")


        # 3D Viewer
        self.viewer = MyViewer(self)
        self.viewer.InitDriver()

        # Create the dock widget and connect it to the viewer
        self.info_dock = EnhancedInfoDock(self)
        self.viewer.info_dock = self.info_dock  # Connect viewer to info panel
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.info_dock)

        # Right panel layout
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.addWidget(self.chat_panel)
        right_layout.addWidget(self.viewer, stretch=1)

        # Action buttons row
        btn_layout = QHBoxLayout()
        for name in ["Heatmap", "Viewport", "Upload Neo4j", "Export"]:
            btn = QPushButton(name)
            btn.setStyleSheet(f"background: {COLOR_INPUT_BG}; color: {COLOR_INPUT_TEXT}; font-weight: bold;")
            if name == "Heatmap":
                btn.clicked.connect(self.viewer.generate_heatmap)
            elif name == "Upload Neo4j":
                btn.clicked.connect(self.upload_to_neo4j)
                btn.clicked.connect(self.show_graph_viewer)
            elif name == "Export":
                btn.clicked.connect(self.export_ifc_to_excel)

            btn_layout.addWidget(btn)  # ✅ this was missing

        right_layout.addLayout(btn_layout)  # ✅ needed to show buttons

        # Enhanced Rendering Controls
        enhanced_group = QGroupBox("Enhanced Rendering")
        enhanced_group.setStyleSheet(f"""
            QGroupBox {{
                background: {BG_PANEL};
                border: 2px solid {BORDER_ACCENT};
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
                color: {TEXT_ACCENT};
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)
        enhanced_layout = QVBoxLayout(enhanced_group)
        
        # Enhanced rendering toggle
        self.enhanced_btn = QPushButton("Enable Enhanced Rendering")
        self.enhanced_btn.setStyleSheet(btn_style)
        self.enhanced_btn.clicked.connect(self.toggle_enhanced_rendering)
        enhanced_layout.addWidget(self.enhanced_btn)
        
        # Apartment overlay toggle
        self.apartment_btn = QPushButton("Toggle Apartment Overlay")
        self.apartment_btn.setStyleSheet(btn_style)
        self.apartment_btn.clicked.connect(self.toggle_apartment_overlay)
        self.apartment_btn.setEnabled(False)  # Disabled until IFC is loaded
        enhanced_layout.addWidget(self.apartment_btn)
        
        # Reset colors button
        self.reset_btn = QPushButton("Reset to Element Colors")
        self.reset_btn.setStyleSheet(btn_style)
        self.reset_btn.clicked.connect(self.reset_element_colors)
        self.reset_btn.setEnabled(False)  # Disabled until IFC is loaded
        enhanced_layout.addWidget(self.reset_btn)
        
        # Debug ML Integration button
        self.debug_ml_btn = QPushButton("Debug ML Integration")
        self.debug_ml_btn.setStyleSheet(btn_style)
        self.debug_ml_btn.clicked.connect(self.debug_ml_integration)
        enhanced_layout.addWidget(self.debug_ml_btn)
        
        # Debug IFC Analysis button
        self.debug_ifc_btn = QPushButton("Debug IFC Analysis")
        self.debug_ifc_btn.setStyleSheet(btn_style)
        self.debug_ifc_btn.clicked.connect(self.debug_ifc_analysis)
        enhanced_layout.addWidget(self.debug_ifc_btn)
        
        # Acoustic Failure Analysis button
        self.acoustic_failure_btn = QPushButton("Analyze Acoustic Failures")
        self.acoustic_failure_btn.setStyleSheet(btn_style)
        self.acoustic_failure_btn.clicked.connect(self.run_acoustic_failure_analysis)
        enhanced_layout.addWidget(self.acoustic_failure_btn)
        
        # Reset Acoustic Overlay button
        self.reset_acoustic_btn = QPushButton("Reset Acoustic Overlay")
        self.reset_acoustic_btn.setStyleSheet(btn_style)
        self.reset_acoustic_btn.clicked.connect(self.reset_acoustic_failure_overlay)
        enhanced_layout.addWidget(self.reset_acoustic_btn)
        
        right_layout.addWidget(enhanced_group)

        # Return to Inputs button
        self.return_to_inputs_btn = QPushButton("Return to Inputs")
        self.return_to_inputs_btn.setStyleSheet(f"background: {COLOR_INPUT_BG}; color: {COLOR_INPUT_TEXT}; font-weight: bold;")
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
        tabs.setStyleSheet(tab_style)

    # def export_ifc_summary(self):
    #     data = self.viewer.extract_model_data()  # Uses your existing method
    #     if not data:
    #         QMessageBox.warning(self, "Export Failed", "No IFC data to export.")
    #         return

    def show_graph_viewer(self):
        if hasattr(self, "graph_viewer"):
            self.graph_viewer.show()
            self.graph_viewer.raise_()
        else:
            self.graph_viewer = GraphViewer(self)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.graph_viewer)
            self.graph_viewer.show()


    def upload_to_neo4j(self):
        data = self.viewer.extract_model_data()
        if not data:
            QMessageBox.warning(self, "Upload Failed", "No IFC data to upload.")
            return
        try:
            conn = Neo4jConnector(password="123456789")
            conn.insert_all(data)
            conn.close()
            QMessageBox.information(self, "Success", "IFC data uploaded to Neo4j.")
        except Exception as e:
            QMessageBox.warning(self, "Neo4j Error", str(e))


    def export_ifc_to_excel(self):
        import os
        import pandas as pd
        from pathlib import Path
        from PyQt6.QtWidgets import QMessageBox

        # Resolve the /Reports folder relative to the script
        reports_path = Path(__file__).resolve().parent.parent / "Reports"
        reports_path.mkdir(parents=True, exist_ok=True)  # Create if missing

        # Generate the default export path
        export_file = reports_path / "ifc_model_export.xlsx"

        # Extract the data
        data = self.viewer.extract_model_data()
        if not data:
            QMessageBox.warning(self, "Export Failed", "No IFC data found.")
            return

        # Convert to DataFrame and export
        try:
            df = pd.DataFrame(data)
            df.to_excel(export_file, index=False)
            QMessageBox.information(self, "Success", f"IFC summary exported to:\n{export_file}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export:\n{e}")

    
    def show_inputs_panel(self):
        self.left_panel.setVisible(True)

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
            # Get comprehensive IFC data analysis
            ifc_analysis = self.get_comprehensive_ifc_analysis()
            
            # Create detailed summary for the AI
            summary = f"""
IFC Model Analysis:
{ifc_analysis}

Current Loaded Elements:
- Total loaded: {len(self.viewer.shape_to_ifc)} elements
- Loaded types: {', '.join(set(elem.is_a() for elem in self.viewer.shape_to_ifc.values()))}

Note: Some IFC elements (like IfcSpace) may not have geometric representation and won't appear in the 3D viewer, but they exist in the IFC data structure.
"""

            prompt = f"{question}\n\n{summary}"

            from server.config import client, completion_model
            response = client.chat.completions.create(
                model=completion_model,
                messages=[
                    {"role": "system", "content": "You are an architectural acoustics advisor with expertise in IFC (Industry Foundation Classes) building models. You can analyze building elements, spaces, and provide acoustic insights. When asked about IFC elements, provide specific counts and details from the data provided. For acoustic analysis questions, focus on identifying specific spaces with acoustic issues based on their properties, surrounding elements, and risk factors. Provide actionable recommendations for improving acoustic performance."},
                    {"role": "user", "content": prompt}
                ]
            )

            answer = response.choices[0].message.content.strip()
            self.chat_history.append(f"<span style='color:#E98801;'><b>AI:</b> {answer}</span>")

        except Exception as e:
            self.chat_history.append(f"<span style='color:red;'>[Error]: {e}</span>")
            print(f"❌ Chatbot error: {e}")
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
            analysis.append("📊 Complete IFC Element Analysis:")
            analysis.append("=" * 40)
            
            # Sort by count (highest first)
            sorted_elements = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)
            
            for elem_type, count in sorted_elements:
                analysis.append(f"• {elem_type}: {count}")
            
            # Enhanced space analysis with acoustic properties
            space_count = element_counts.get('IfcSpace', 0)
            if space_count > 0:
                analysis.append(f"\n🏢 Enhanced Spaces Analysis:")
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
            
            analysis.append(f"\n📈 Summary Statistics (first 10 spaces):")
            analysis.append(f"• Average volume: {avg_volume:.1f} m³")
            analysis.append(f"• Total spaces analyzed: {min(len(spaces), 10)}")
            
            return analysis
            
        except Exception as e:
            return [f"Error in detailed space analysis: {e}"]

    def analyze_single_space(self, space, ifc_file):
        """Analyze a single space with its acoustic properties and relationships"""
        try:
            space_info = {
                'name': getattr(space, 'Name', 'Unnamed'),
                'global_id': getattr(space, 'GlobalId', 'Unknown'),
                'volume': 0,
                'area': 0,
                'height': 0,
                'acoustic_properties': {},
                'surrounding_elements': {},
                'acoustic_risk': 'Unknown'
            }
            
            # Extract geometric properties
            if hasattr(space, 'Representation') and space.Representation:
                try:
                    # Try to get volume from representation
                    shape = ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(), space)
                    if shape and hasattr(shape, 'geometry'):
                        # Calculate volume (simplified)
                        bbox = shape.geometry.BoundingBox()
                        if bbox:
                            width = bbox.Xmax() - bbox.Xmin()
                            height = bbox.Ymax() - bbox.Ymin()
                            depth = bbox.Zmax() - bbox.Zmin()
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
                                    prop_name = prop.Name.lower()
                                    prop_value = getattr(prop.NominalValue, 'wrappedValue', None)
                                    
                                    # Look for acoustic-related properties
                                    if any(acoustic_term in prop_name for acoustic_term in ['rt60', 'spl', 'acoustic', 'sound', 'noise', 'reverberation']):
                                        space_info['acoustic_properties'][prop.Name] = prop_value
            
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
                    # Enable apartment overlay controls
                    self.apartment_btn.setEnabled(True)
                    self.reset_btn.setEnabled(True)
                    
                    # Show success message
                    self.chat_history.append(f"✅ IFC file loaded: {os.path.basename(file_path)}")
                    
                    # Show model statistics
                    summary = self.viewer.get_ifc_summary()
                    if summary:
                        stats_text = f"📊 Model Statistics:\n"
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
        # 1. Get user inputs
        user_input = {
            "Apartment_Type": self.geo_apartment.currentText(),
            "Zone": "HD-Urban-V0",  # Default zone
            "wall_material": self.geo_wall.currentText(),
            "window_material": self.geo_window.currentText(),
            "Floor_Level": 2,  # Replace with an actual floor input if available
            "activity": self.activity_box.currentText(),
        }

        # 2. Get geometry data from InfoDock if available
        geometry_data = None
        if hasattr(self, 'info_dock') and self.info_dock and hasattr(self.info_dock, 'current_element_data'):
            geometry_data = self.info_dock.current_element_data
            if geometry_data:
                print("🎯 Using geometry data for enhanced ML prediction")

        # 3. Build a natural language prompt
        user_question = (
            f"Evaluate acoustic comfort for a {user_input['Apartment_Type']} apartment "
            f"in zone {user_input['Zone']}, with {user_input['wall_material']} walls and "
            f"{user_input['window_material']} windows. Activity is {user_input['activity']}."
        )

        # 4. Show user input in chat panel
        self.chat_history.append(f"<b>You:</b> {user_question}")

        # 5. Call enhanced ML pipeline with geometry data
        try:
            from scripts.core.acoustic_pipeline import run_pipeline

            result = run_pipeline(user_input, user_question, geometry_data)
            print(f"🎯 Pipeline result: {result}")
            
            # Handle enhanced results
            if result.get("enhanced") and isinstance(result.get("result"), dict):
                enhanced_result = result["result"]
                
                # Display SQL results first
                if "sql_result" in enhanced_result:
                    sql_result = enhanced_result["sql_result"]
                    if isinstance(sql_result, dict) and "summary" in sql_result:
                        self.chat_history.append(f"<span style='color:#2196F3;'><b>📊 SQL Analysis:</b> {sql_result['summary']}</span>")
                    elif isinstance(sql_result, str):
                        self.chat_history.append(f"<span style='color:#2196F3;'><b>📊 SQL Analysis:</b> {sql_result}</span>")
                
                # Display comfort prediction
                comfort_pred = enhanced_result.get("comfort_prediction")
                if comfort_pred and "error" not in comfort_pred:
                    comfort_msg = f"🎯 <b>ML Comfort Prediction:</b> {comfort_pred['comfort_score']} (Confidence: {comfort_pred['confidence']:.1%})"
                    self.chat_history.append(f"<span style='color:#4CAF50;'>{comfort_msg}</span>")
                elif comfort_pred and "error" in comfort_pred:
                    self.chat_history.append(f"<span style='color:#FF9800;'>⚠️ ML Prediction: {comfort_pred['error']}</span>")
                
                # Display recommendations
                recommendations = enhanced_result.get("recommendations", [])
                if recommendations:
                    rec_msg = "<b>💡 ML Recommendations:</b><br>"
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
                # Standard result
                summary = result.get("summary") or result.get("response") or "No summary returned."
                self.chat_history.append(f"<span style='color:#E98801;'><b>AI:</b> {summary}</span>")

        except Exception as e:
            self.chat_history.append(f"<span style='color:red;'>[Error]: {e}</span>")
            print(f"❌ Evaluation error: {e}")
            import traceback
            traceback.print_exc()

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
                    print("⚠️ Selected shape not found in shape_to_ifc mapping")
                    self.show_ifc_panel(None)
            elif not ais_shape:
                if self.last_selected is not None:
                    print("🔍 No element selected - clearing InfoDock")
                    self.last_selected = None
                    self.show_ifc_panel(None)
                    
        except Exception as e:
            print(f"❌ Error in check_occ_selection: {e}")

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
            print(f"❌ Error showing IFC panel: {e}")
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
                self.chat_history.append(f"<span style='color:#9C27B0;'><b>🎵 Space Acoustic Analysis: {space_info['name']}</b></span>")
                
                # Basic info
                self.chat_history.append(f"<span style='color:#2196F3;'>📏 Volume: {space_info['volume']:.1f} m³ | Area: {space_info['area']:.1f} m² | Height: {space_info['height']:.1f} m</span>")
                
                # Acoustic properties
                if space_info['acoustic_properties']:
                    self.chat_history.append(f"<span style='color:#4CAF50;'>🎵 Acoustic Properties:</span>")
                    for prop, value in space_info['acoustic_properties'].items():
                        self.chat_history.append(f"<span style='color:#4CAF50;'>   • {prop}: {value}</span>")
                
                # Surrounding elements
                if space_info['surrounding_elements']:
                    self.chat_history.append(f"<span style='color:#FF9800;'>🏗️ Surrounding Elements:</span>")
                    for elem_type, count in space_info['surrounding_elements'].items():
                        self.chat_history.append(f"<span style='color:#FF9800;'>   • {elem_type}: {count}</span>")
                
                # Acoustic status
                if failures:
                    self.chat_history.append(f"<span style='color:#F44336;'>❌ Acoustic Status: {severity.upper()} RISK</span>")
                    self.chat_history.append(f"<span style='color:#F44336;'>🚨 Failures:</span>")
                    for failure in failures:
                        self.chat_history.append(f"<span style='color:#F44336;'>   • {failure}</span>")
                    
                    # Recommendations
                    recommendations = self.get_acoustic_recommendations(space_info, failures)
                    if recommendations:
                        self.chat_history.append(f"<span style='color:#4CAF50;'>💡 Recommendations:</span>")
                        for rec in recommendations[:3]:  # Show top 3
                            self.chat_history.append(f"<span style='color:#4CAF50;'>   • {rec}</span>")
                else:
                    self.chat_history.append(f"<span style='color:#4CAF50;'>✅ Acoustic Status: PASS - No issues detected</span>")
                
                # Add separator
                self.chat_history.append("<hr style='border-color:#FF9500; margin:10px 0;'>")
                
        except Exception as e:
            print(f"❌ Error showing space acoustic info: {e}")

    def run_ml_analysis_for_element(self, element_data):
        """Run ML analysis for a specific element and display results in chat"""
        try:
            if not element_data:
                self.chat_history.append("<span style='color:red;'>❌ No element data available for ML analysis</span>")
                return
            
            element_type = element_data.get('type', 'Unknown')
            element_name = element_data.get('name', 'Unnamed')
            
            self.chat_history.append(f"<b>🤖 Running ML Analysis for: {element_type} - {element_name}</b>")
            
            # Import and run ML analysis
            from scripts.core.geometry_ml_interface import geometry_ml_interface
            
            if geometry_ml_interface is None:
                self.chat_history.append("<span style='color:red;'>❌ ML interface not available</span>")
                return
            
            # Extract data for ML processing
            extracted_data = geometry_ml_interface.extract_element_data(element_data)
            if not extracted_data:
                self.chat_history.append("<span style='color:red;'>❌ Could not extract data for ML processing</span>")
                return
            
            self.chat_history.append(f"<span style='color:#2196F3;'>🔍 Extracted data: {extracted_data.get('element_type', 'Unknown')}</span>")
            
            # Run comfort prediction
            comfort_result = geometry_ml_interface.predict_comfort_for_element(extracted_data)
            
            if "error" in comfort_result:
                self.chat_history.append(f"<span style='color:red;'>❌ ML Prediction Error: {comfort_result['error']}</span>")
            else:
                # Display comfort prediction results
                comfort_score = comfort_result.get('comfort_score', 'N/A')
                confidence = comfort_result.get('confidence', 0)
                self.chat_history.append(f"<span style='color:#4CAF50;'>🎯 Comfort Score: {comfort_score} (Confidence: {confidence:.1%})</span>")
                
                # Display key features used
                features_used = comfort_result.get('features_used', {})
                if features_used:
                    feature_text = "📊 Features used: "
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
                rec_text = "💡 Recommendations:<br>"
                for rec in recommendations:
                    rec_text += f"• {rec}<br>"
                self.chat_history.append(f"<span style='color:#FF9800;'>{rec_text}</span>")
            else:
                self.chat_history.append("<span style='color:#FF9800;'>💡 No specific recommendations available</span>")
            
            # Add separator
            self.chat_history.append("<hr style='border-color:#FF9500; margin:10px 0;'>")
            
        except Exception as e:
            error_msg = f"❌ ML analysis failed: {str(e)}"
            self.chat_history.append(f"<span style='color:red;'>{error_msg}</span>")
            print(error_msg)
            import traceback
            traceback.print_exc()

    def reset_element_colors(self):
        """Reset colors to original element type colors"""
        self.viewer.reset_to_element_colors()
        self.apartment_btn.setText("Toggle Apartment Overlay")
        self.apartment_btn.setStyleSheet(btn_style)
        self.chat_history.append("✅ Reset to element type colors")

    def debug_ifc_file(self):
        """Debug IFC file contents to help diagnose loading issues"""
        if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
            QMessageBox.information(self, "Debug", "No IFC file loaded. Please load an IFC file first.")
            return
            
        try:
            element_counts = self.viewer.analyze_ifc_file(self.viewer.current_ifc_path)
            
            # Create detailed debug report
            debug_text = f"🔍 IFC Debug Report: {os.path.basename(self.viewer.current_ifc_path)}\n"
            debug_text += "=" * 50 + "\n"
            
            if element_counts:
                debug_text += "📊 Element Analysis:\n"
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
                    debug_text += "⚠️ No geometric elements found!\n"
                    debug_text += "💡 This IFC file may not contain loadable geometry.\n"
                    debug_text += "💡 Try a different IFC file with walls, slabs, or other building elements.\n"
                else:
                    debug_text += "✅ Geometric elements found - should load successfully.\n"
                
                # Check loaded elements
                loaded_count = len(self.viewer.shape_to_ifc)
                debug_text += f"\n📦 Currently loaded: {loaded_count} elements\n"
                
                if loaded_count == 0 and geometric_count > 0:
                    debug_text += "⚠️ Elements found but not loaded - there may be a loading issue.\n"
                elif loaded_count > 0:
                    debug_text += "✅ Elements loaded successfully.\n"
            
            # Show in chat
            self.chat_history.append(debug_text)
            
            # Also show in a dialog for better visibility
            QMessageBox.information(self, "IFC Debug Report", debug_text)
            
        except Exception as e:
            error_msg = f"❌ Debug error: {str(e)}"
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
            print(f"❌ Error analyzing IFC file: {e}")
            return None

    def debug_ml_integration(self):
        """Debug ML integration to see what data is available"""
        try:
            self.chat_history.append("<b>🔍 ML Integration Debug Report</b>")
            
            # Check if InfoDock has data
            if hasattr(self, 'info_dock') and self.info_dock:
                if hasattr(self.info_dock, 'current_element_data') and self.info_dock.current_element_data:
                    data = self.info_dock.current_element_data
                    self.chat_history.append(f"✅ InfoDock has element data: {data.get('type', 'Unknown')}")
                    self.chat_history.append(f"📋 Element name: {data.get('name', 'N/A')}")
                    self.chat_history.append(f"🎵 RT60: {data.get('acoustic_properties', {}).get('RT60', 'N/A')}")
                    self.chat_history.append(f"🎵 SPL: {data.get('acoustic_properties', {}).get('SPL', 'N/A')}")
                    self.chat_history.append(f"📏 Area: {data.get('dimensions', {}).get('Gross Planned Area', 'N/A')}")
                    
                    # Test ML prediction with current data
                    self.test_ml_prediction(data)
                else:
                    self.chat_history.append("❌ InfoDock has no element data")
            else:
                self.chat_history.append("❌ InfoDock not available")
            
            # Check viewer data
            if hasattr(self, 'viewer') and self.viewer:
                loaded_elements = len(self.viewer.shape_to_ifc)
                self.chat_history.append(f"📦 Viewer has {loaded_elements} loaded elements")
                
                # Show element types
                element_types = {}
                for ifc_elem in self.viewer.shape_to_ifc.values():
                    elem_type = ifc_elem.is_a()
                    element_types[elem_type] = element_types.get(elem_type, 0) + 1
                
                self.chat_history.append("📊 Element types loaded:")
                for elem_type, count in element_types.items():
                    self.chat_history.append(f"   • {elem_type}: {count}")
            else:
                self.chat_history.append("❌ Viewer not available")
            
            # Check ML models
            try:
                from scripts.core.geometry_ml_interface import GeometryMLInterface
                ml_interface = GeometryMLInterface()
                if ml_interface.comfort_model:
                    self.chat_history.append("✅ Comfort model loaded")
                else:
                    self.chat_history.append("❌ Comfort model not loaded")
                if ml_interface.acoustic_model:
                    self.chat_history.append("✅ Acoustic model loaded")
                else:
                    self.chat_history.append("❌ Acoustic model not loaded")
            except Exception as e:
                self.chat_history.append(f"❌ ML interface error: {e}")
            
        except Exception as e:
            self.chat_history.append(f"❌ Debug error: {e}")

    def debug_ifc_analysis(self):
        """Debug IFC analysis to see what data is available"""
        try:
            self.chat_history.append("<b>🔍 IFC Analysis Debug Report</b>")
            
            # Check if IFC file is loaded
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                self.chat_history.append("❌ No IFC file loaded")
                return
            
            self.chat_history.append(f"📁 IFC File: {os.path.basename(self.viewer.current_ifc_path)}")
            
            # Get comprehensive analysis
            analysis = self.get_comprehensive_ifc_analysis()
            self.chat_history.append(f"📊 Analysis Results:")
            
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
                self.chat_history.append(f"❌ AI test failed: {e}")
            
        except Exception as e:
            self.chat_history.append(f"❌ IFC Analysis debug error: {e}")
            import traceback
            traceback.print_exc()

    def test_ml_prediction(self, element_data):
        """Test ML prediction with the given element data"""
        try:
            from scripts.core.geometry_ml_interface import GeometryMLInterface
            ml_interface = GeometryMLInterface()
            
            # Extract data for ML processing
            extracted_data = ml_interface.extract_element_data(element_data)
            self.chat_history.append(f"🔍 Extracted ML data: {extracted_data}")
            
            # Test comfort prediction
            comfort_result = ml_interface.predict_comfort_for_element(extracted_data)
            if "error" not in comfort_result:
                self.chat_history.append(f"🎯 Test ML Prediction: {comfort_result['comfort_score']} (Confidence: {comfort_result['confidence']:.1%})")
            else:
                self.chat_history.append(f"❌ ML Prediction Error: {comfort_result['error']}")
            
            # Test recommendations
            recommendations = ml_interface.get_ml_recommendations(extracted_data)
            if recommendations:
                self.chat_history.append(f"💡 Test Recommendations: {len(recommendations)} found")
                for rec in recommendations[:2]:  # Show first 2
                    self.chat_history.append(f"   • {rec}")
            else:
                self.chat_history.append("❌ No recommendations generated")
                
        except Exception as e:
            self.chat_history.append(f"❌ ML test failed: {e}")

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
            print("✅ Enhanced rendering enabled")
        else:
            self.viewer.enhanced_rendering_enabled = False
            self.enhanced_btn.setText("Enable Enhanced Rendering")
            self.enhanced_btn.setStyleSheet(btn_style)
            print("✅ Enhanced rendering disabled")

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
                stats_text = f"🏢 Apartment Overlay Applied:\n"
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
            self.chat_history.append("✅ Reset to element type colors")

    def reset_element_colors(self):
        """Reset colors to original element type colors"""
        self.viewer.reset_to_element_colors()
        self.apartment_btn.setText("Toggle Apartment Overlay")
        self.apartment_btn.setStyleSheet(btn_style)
        self.chat_history.append("✅ Reset to element type colors")

    def identify_failing_acoustic_spaces(self):
        """Identify spaces that are failing acoustic requirements"""
        try:
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                return "No IFC file loaded. Please load an IFC file first."
            
            ifc_file = ifcopenshell.open(self.viewer.current_ifc_path)
            spaces = ifc_file.by_type("IfcSpace")
            
            failing_spaces = []
            analysis_results = []
            
            analysis_results.append("🔍 Acoustic Failure Analysis:")
            analysis_results.append("=" * 40)
            
            for i, space in enumerate(spaces):
                space_info = self.analyze_single_space(space, ifc_file)
                if space_info:
                    # Check for acoustic failures
                    failures = self.check_acoustic_failures(space_info)
                    if failures:
                        failing_spaces.append({
                            'space': space_info,
                            'failures': failures
                        })
                        
                        analysis_results.append(f"\n❌ FAILING SPACE {i+1}: {space_info['name']}")
                        analysis_results.append(f"   • Global ID: {space_info['global_id']}")
                        analysis_results.append(f"   • Volume: {space_info['volume']:.1f} m³")
                        analysis_results.append(f"   • Area: {space_info['area']:.1f} m²")
                        
                        for failure in failures:
                            analysis_results.append(f"   • FAILURE: {failure}")
                        
                        # Add recommendations
                        recommendations = self.get_acoustic_recommendations(space_info, failures)
                        if recommendations:
                            analysis_results.append(f"   • RECOMMENDATIONS:")
                            for rec in recommendations:
                                analysis_results.append(f"     - {rec}")
            
            if not failing_spaces:
                analysis_results.append("\n✅ No acoustic failures detected in the analyzed spaces.")
                analysis_results.append("All spaces appear to meet basic acoustic requirements.")
            else:
                analysis_results.append(f"\n📊 Summary:")
                analysis_results.append(f"• Total spaces analyzed: {len(spaces)}")
                analysis_results.append(f"• Spaces with acoustic failures: {len(failing_spaces)}")
                analysis_results.append(f"• Failure rate: {(len(failing_spaces)/len(spaces)*100):.1f}%")
            
            return "\n".join(analysis_results)
            
        except Exception as e:
            return f"Error in acoustic failure analysis: {e}"

    def check_acoustic_failures(self, space_info):
        """Check for specific acoustic failures in a space"""
        failures = []
        
        try:
            # Volume-based failures
            volume = space_info.get('volume', 0)
            if volume > 2000:
                failures.append("Excessive volume (>2000 m³) - may cause poor speech intelligibility")
            elif volume < 30:
                failures.append("Very small volume (<30 m³) - may cause sound pressure issues")
            
            # Area-based failures
            area = space_info.get('area', 0)
            if area > 500:
                failures.append("Large floor area (>500 m²) - may require acoustic treatment")
            
            # Height-based failures
            height = space_info.get('height', 0)
            if height > 5:
                failures.append("High ceiling (>5m) - may cause excessive reverberation")
            elif height < 2.4:
                failures.append("Low ceiling (<2.4m) - may cause sound pressure issues")
            
            # Surrounding elements failures
            surrounding = space_info.get('surrounding_elements', {})
            wall_count = surrounding.get('IfcWall', 0)
            door_count = surrounding.get('IfcDoor', 0)
            window_count = surrounding.get('IfcWindow', 0)
            
            if wall_count < 3:
                failures.append("Insufficient walls (<3) - poor acoustic isolation")
            if door_count > 5:
                failures.append("Too many doors (>5) - excessive sound leakage")
            if window_count > 4:
                failures.append("Too many windows (>4) - poor sound insulation")
            
            # Acoustic properties failures
            acoustic_props = space_info.get('acoustic_properties', {})
            if not acoustic_props:
                failures.append("No acoustic properties defined - cannot assess performance")
            else:
                for prop_name, value in acoustic_props.items():
                    if 'rt60' in prop_name.lower() and value:
                        try:
                            rt60_val = float(value)
                            if rt60_val > 3.0:
                                failures.append(f"Excessive RT60 ({rt60_val}s) - poor acoustic performance")
                            elif rt60_val < 0.2:
                                failures.append(f"Very low RT60 ({rt60_val}s) - space may be too dead")
                        except:
                            pass
                    
                    if 'spl' in prop_name.lower() and value:
                        try:
                            spl_val = float(value)
                            if spl_val > 70:
                                failures.append(f"High SPL ({spl_val} dBA) - excessive noise levels")
                        except:
                            pass
            
            return failures
            
        except Exception as e:
            return [f"Error checking failures: {e}"]

    def get_acoustic_recommendations(self, space_info, failures):
        """Get specific recommendations for improving acoustic performance"""
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

    def run_acoustic_failure_analysis(self):
        """Run acoustic failure analysis and display results in chat"""
        try:
            result = self.identify_failing_acoustic_spaces()
            self.chat_history.append(f"<span style='color:#FF9500;'><b>🔍 Acoustic Failure Analysis:</b></span>")
            self.chat_history.append(f"<span style='color:#FDF6F6;'>{result}</span>")
            
            # Create visual overlay for failing spaces
            self.create_acoustic_failure_overlay()
            
        except Exception as e:
            self.chat_history.append(f"<span style='color:red;'>[Error]: {e}</span>")
            print(f"❌ Acoustic failure analysis failed: {e}")
            import traceback
            traceback.print_exc()

    def create_acoustic_failure_overlay(self):
        """Create a visual overlay highlighting failing acoustic spaces"""
        try:
            if not hasattr(self.viewer, 'current_ifc_path') or not self.viewer.current_ifc_path:
                self.chat_history.append("<span style='color:red;'>❌ No IFC file loaded for overlay</span>")
                return
            
            self.chat_history.append("<span style='color:#4CAF50;'><b>🎨 Creating Acoustic Failure Overlay...</b></span>")
            
            # Get failing spaces
            failing_spaces = self.get_failing_spaces_data()
            
            if not failing_spaces:
                self.chat_history.append("<span style='color:#FF9800;'>✅ No failing spaces found - all spaces meet acoustic requirements</span>")
                return
            
            # Apply color coding to the viewer
            self.apply_acoustic_failure_colors(failing_spaces)
            
            # Show overlay legend
            self.show_acoustic_failure_legend(failing_spaces)
            
            self.chat_history.append(f"<span style='color:#4CAF50;'>✅ Acoustic failure overlay applied! {len(failing_spaces)} failing spaces highlighted</span>")
            
        except Exception as e:
            self.chat_history.append(f"<span style='color:red;'>❌ Overlay creation failed: {e}</span>")
            print(f"❌ Overlay error: {e}")
            import traceback
            traceback.print_exc()

    def get_failing_spaces_data(self):
        """Get data for all failing acoustic spaces"""
        try:
            ifc_file = ifcopenshell.open(self.viewer.current_ifc_path)
            spaces = ifc_file.by_type("IfcSpace")
            
            failing_spaces = []
            
            for space in spaces:
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
            
            return failing_spaces
            
        except Exception as e:
            print(f"Error getting failing spaces data: {e}")
            return []

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
        """Get color based on severity level"""
        color_map = {
            "critical": (1.0, 0.0, 0.0),    # Bright red
            "high": (1.0, 0.5, 0.0),        # Orange
            "medium": (1.0, 1.0, 0.0),      # Yellow
            "low": (0.0, 1.0, 0.0)          # Green
        }
        return color_map.get(severity, (1.0, 0.5, 0.0))

    def apply_acoustic_failure_colors(self, failing_spaces):
        """Apply color coding to failing spaces in the viewer"""
        try:
            colored_count = 0
            
            # Create mapping of space Global IDs to failure data
            space_failure_map = {fs['global_id']: fs for fs in failing_spaces}
            
            # Apply colors to spaces in the viewer
            for ais_shape, ifc_elem in self.viewer.shape_to_ifc.items():
                if ifc_elem.is_a() == "IfcSpace":
                    global_id = getattr(ifc_elem, "GlobalId", "")
                    
                    if global_id in space_failure_map:
                        failure_data = space_failure_map[global_id]
                        color = failure_data['color']
                        
                        # Apply color to the shape
                        color_obj = Quantity_Color(color[0], color[1], color[2], Quantity_TOC_RGB)
                        self.viewer._display.Context.SetColor(ais_shape, color_obj, False)
                        
                        # Make it more visible
                        self.viewer._display.Context.SetTransparency(ais_shape, 0.3, False)
                        
                        colored_count += 1
            
            # Update the display
            self.viewer._display.Context.UpdateCurrentViewer()
            self.viewer._display.Repaint()
            
            print(f"✅ Applied acoustic failure colors to {colored_count} spaces")
            
        except Exception as e:
            print(f"❌ Error applying acoustic failure colors: {e}")

    def show_acoustic_failure_legend(self, failing_spaces):
        """Show legend for acoustic failure overlay"""
        try:
            # Count by severity
            severity_counts = {}
            for fs in failing_spaces:
                severity = fs['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Create legend HTML
            legend_html = """
            <div style="background: #2C2C2C; padding: 15px; border-radius: 10px; color: white; margin: 10px 0;">
                <h3 style="margin-top: 0; color: #FF9500;">🎵 Acoustic Failure Overlay</h3>
                <div style="display: flex; flex-direction: column; gap: 8px;">
            """
            
            severity_info = {
                "critical": {"color": "rgb(255, 0, 0)", "desc": "Critical - Multiple severe issues"},
                "high": {"color": "rgb(255, 128, 0)", "desc": "High - Significant acoustic problems"},
                "medium": {"color": "rgb(255, 255, 0)", "desc": "Medium - Moderate issues"},
                "low": {"color": "rgb(0, 255, 0)", "desc": "Low - Minor issues"}
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
                <div style="margin-top: 10px; font-size: 12px; color: #CCC;">
                    💡 Hover over colored spaces to see detailed failure information
                </div>
            </div>
            """
            
            # Display legend in chat
            self.chat_history.append(f"<span style='color:#4CAF50;'><b>📊 Acoustic Failure Summary:</b></span>")
            for severity, count in severity_counts.items():
                self.chat_history.append(f"<span style='color:#FF9500;'>• {severity.title()}: {count} spaces</span>")
            
        except Exception as e:
            print(f"❌ Error showing legend: {e}")

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
            
            self.chat_history.append("<span style='color:#4CAF50;'>✅ Acoustic failure overlay reset</span>")
            
        except Exception as e:
            print(f"❌ Error resetting overlay: {e}")

if __name__ == "__main__":
        app = QApplication(sys.argv)
        win = EcoformMainWindow()
        win.show()
        sys.exit(app.exec())