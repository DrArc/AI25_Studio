import sys
import os
import json
import pandas as pd
from PyQt6.QtWidgets import QDockWidget, QTextBrowser, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import ML interface
try:
    from .geometry_ml_interface import geometry_ml_interface
    ML_AVAILABLE = True
    print("✅ ML interface loaded successfully")
except ImportError as e:
    print(f"⚠️ ML interface not available: {e}")
    ML_AVAILABLE = False
    geometry_ml_interface = None

class EnhancedInfoDock(QDockWidget):
    """
    Enhanced InfoDock that displays IFC element information and integrates with ML pipeline
    """
    
    # Signal to notify when element data is available for ML
    element_data_ready = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__("Enhanced IFC Element Info", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.setMinimumWidth(400)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create text browser for content
        self.browser = QTextBrowser()
        self.browser.setStyleSheet("""
            QTextBrowser {
                background-color: #1A1A1A;
                color: #FDF6F6;
                font-family: 'Segoe UI';
                font-size: 12px;
                padding: 10px;
                border: 1px solid #FF9500;
                border-radius: 8px;
            }
        """)
        layout.addWidget(self.browser)
        
        # Create button layout
        button_layout = QHBoxLayout()
        
        # ML Analysis button
        self.ml_btn = QPushButton("🤖 Run ML Analysis")
        self.ml_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF9500, stop:1 #FF7F50);
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #FFA040;
            }
            QPushButton:disabled {
                background: #666666;
                color: #999999;
            }
        """)
        self.ml_btn.clicked.connect(self.run_ml_analysis)
        self.ml_btn.setEnabled(False)  # Disabled until element is selected
        button_layout.addWidget(self.ml_btn)
        
        # Clear button
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background: #666666;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #777777;
            }
        """)
        self.clear_btn.clicked.connect(self.clear_content)
        button_layout.addWidget(self.clear_btn)
        
        layout.addLayout(button_layout)
        
        # Initialize data storage
        self.current_element_data = None
        self.parsed_data = {}
        self.apartment_overlay_active = False
        
        # Connect signal to ML pipeline
        self.element_data_ready.connect(self.on_element_data_ready)
        
        self.setFloating(False)
        self.hide()  # Start hidden

    def load_parsed_data(self, ifc_file_path):
        """Load parsed CSV/JSON data for enhanced element information"""
        try:
            # Extract base path from IFC file
            base_path = os.path.dirname(ifc_file_path)
            base_name = os.path.splitext(os.path.basename(ifc_file_path))[0]
            
            # Look for parsed data files
            json_path = os.path.join(base_path, "parse", f"{base_name}.json")
            csv_path = os.path.join(base_path, "parse", f"{base_name}.csv")
            
            self.parsed_data = {}
            
            # Load JSON data if available
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    if isinstance(json_data, list):
                        for item in json_data:
                            if 'GlobalId' in item:
                                self.parsed_data[item['GlobalId']] = item
                    elif isinstance(json_data, dict):
                        self.parsed_data = json_data
                print(f"✅ Loaded JSON data: {len(self.parsed_data)} elements")
            
            # Load CSV data if available
            if os.path.exists(csv_path):
                try:
                    csv_data = pd.read_csv(csv_path)
                    for _, row in csv_data.iterrows():
                        if 'GlobalId' in row:
                            self.parsed_data[row['GlobalId']] = row.to_dict()
                    print(f"✅ Loaded CSV data: {len(csv_data)} elements")
                except Exception as e:
                    print(f"⚠️ Error loading CSV: {e}")
            
            print(f"📊 Total parsed data loaded: {len(self.parsed_data)} elements")
            
        except Exception as e:
            print(f"❌ Error loading parsed data: {e}")

    def find_element_data(self, global_id):
        """Find additional data for an element by GlobalId"""
        return self.parsed_data.get(global_id, {})

    def format_property_value(self, value):
        """Format property values for display"""
        if value is None:
            return "N/A"
        elif isinstance(value, (int, float)):
            return f"{value:.2f}" if isinstance(value, float) else str(value)
        else:
            return str(value)

    def update_content(self, ifc_elem):
        """Update the InfoDock content with IFC element information"""
        try:
            if ifc_elem is None:
                # Show apartment legend if overlay is active, otherwise show nothing
                if self.apartment_overlay_active:
                    self.browser.setHtml(self.get_apartment_legend_html())
                    self.show()
                else:
                    self.hide()
                self.current_element_data = None
                self.ml_btn.setEnabled(False)
                return
            
            # Extract comprehensive element data
            element_data = self.extract_element_data_for_ml(ifc_elem)
            self.current_element_data = element_data
            
            # Enable ML button if we have valid data
            if element_data and element_data.get('type'):
                self.ml_btn.setEnabled(True)
                # Emit signal for ML pipeline
                self.element_data_ready.emit(element_data)
            
            # Build HTML content
            html_parts = []
            
            # Header with element type and name
            element_type = ifc_elem.is_a()
            element_name = getattr(ifc_elem, "LongName", "") or getattr(ifc_elem, "Name", "") or "Unnamed"
            
            html_parts.append(f"""
                <div style="background: linear-gradient(135deg, #FF9500, #FF7F50); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <h2 style="margin: 0; color: white; font-size: 18px;">🏗️ {element_type}</h2>
                    <p style="margin: 5px 0 0 0; color: white; font-size: 14px;">{element_name}</p>
                </div>
            """)
            
            # Basic Information Section
            html_parts.append("<h3 style='color:#FF9500; border-bottom: 2px solid #FF9500; padding-bottom: 5px;'>📋 Basic Information</h3>")
            
            basic_info = []
            if hasattr(ifc_elem, "GlobalId"):
                basic_info.append(f"<b>Global ID:</b> {ifc_elem.GlobalId}")
            if hasattr(ifc_elem, "Description") and ifc_elem.Description:
                basic_info.append(f"<b>Description:</b> {ifc_elem.Description}")
            if hasattr(ifc_elem, "ObjectType") and ifc_elem.ObjectType:
                basic_info.append(f"<b>Object Type:</b> {ifc_elem.ObjectType}")
            
            # Add apartment info if available
            apartment_info = self.parse_apartment_info(element_name)
            if apartment_info:
                basic_info.append(f"<b>Floor Level:</b> {apartment_info.get('floor', 'N/A')}")
                basic_info.append(f"<b>Apartment Type:</b> {apartment_info.get('type', 'N/A')}")
                basic_info.append(f"<b>Room Number:</b> {apartment_info.get('room', 'N/A')}")
            
            html_parts.append("<div style='margin-bottom: 15px;'>")
            for info in basic_info:
                html_parts.append(f"<p style='margin: 5px 0;'>{info}</p>")
            html_parts.append("</div>")
            
            # Acoustic Properties Section
            acoustic_props = {}
            if hasattr(ifc_elem, "IsDefinedBy"):
                for rel in ifc_elem.IsDefinedBy:
                    if rel.is_a("IfcRelDefinesByProperties"):
                        props = rel.RelatingPropertyDefinition
                        if props.is_a("IfcPropertySet"):
                            for prop in props.HasProperties:
                                pname = prop.Name
                                val = getattr(getattr(prop, "NominalValue", None), "wrappedValue", None)
                                if "RT60" in pname or "SPL" in pname or "acoustic" in pname.lower():
                                    acoustic_props[pname] = val
            
            # Add acoustic properties from parsed data
            global_id = getattr(ifc_elem, 'GlobalId', '')
            if global_id:
                parsed_data = self.find_element_data(global_id)
                for key, value in parsed_data.items():
                    if key.startswith('Property_') and ('RT60' in key or 'SPL' in key or 'acoustic' in key.lower()):
                        prop_name = key.replace('Property_', '')
                        acoustic_props[prop_name] = value
            
            if acoustic_props:
                html_parts.append("<h3 style='color:#4CAF50; border-bottom: 2px solid #4CAF50; padding-bottom: 5px;'>🎵 Acoustic Properties</h3>")
                html_parts.append("<div style='margin-bottom: 15px;'>")
                for prop_name, value in acoustic_props.items():
                    formatted_value = self.format_property_value(value)
                    html_parts.append(f"<p style='margin: 5px 0;'><b>{prop_name}:</b> {formatted_value}</p>")
                html_parts.append("</div>")
            
            # Dimensions Section
            dimensions = {}
            if hasattr(ifc_elem, "IsDefinedBy"):
                for rel in ifc_elem.IsDefinedBy:
                    if rel.is_a("IfcRelDefinesByProperties"):
                        props = rel.RelatingPropertyDefinition
                        if props.is_a("IfcPropertySet"):
                            for prop in props.HasProperties:
                                pname = prop.Name
                                val = getattr(getattr(prop, "NominalValue", None), "wrappedValue", None)
                                if "area" in pname.lower() or "volume" in pname.lower() or "height" in pname.lower() or "width" in pname.lower() or "length" in pname.lower():
                                    dimensions[pname] = val
            
            # Add dimensions from parsed data
            if global_id:
                parsed_data = self.find_element_data(global_id)
                for key, value in parsed_data.items():
                    if key.startswith('Property_') and ('Area' in key or 'Volume' in key or 'Height' in key or 'Width' in key or 'Length' in key or 'GrossPlannedArea' in key):
                        prop_name = key.replace('Property_', '')
                        dimensions[prop_name] = value
            
            if dimensions:
                html_parts.append("<h3 style='color:#2196F3; border-bottom: 2px solid #2196F3; padding-bottom: 5px;'>📏 Dimensions</h3>")
                html_parts.append("<div style='margin-bottom: 15px;'>")
                for prop_name, value in dimensions.items():
                    formatted_value = self.format_property_value(value)
                    html_parts.append(f"<p style='margin: 5px 0;'><b>{prop_name}:</b> {formatted_value}</p>")
                html_parts.append("</div>")
            
            # Materials Section
            materials = []
            if hasattr(ifc_elem, "HasAssociations"):
                for assoc in ifc_elem.HasAssociations:
                    if assoc.is_a("IfcRelAssociatesMaterial"):
                        mat = assoc.RelatingMaterial
                        if hasattr(mat, "Name"):
                            materials.append(mat.Name)
                        elif hasattr(mat, "ForLayerSet"):
                            layers = mat.ForLayerSet.MaterialLayers
                            for layer in layers:
                                if hasattr(layer.Material, "Name"):
                                    materials.append(layer.Material.Name)
            
            if materials:
                html_parts.append("<h3 style='color:#9C27B0; border-bottom: 2px solid #9C27B0; padding-bottom: 5px;'>🏗️ Materials</h3>")
                html_parts.append("<div style='margin-bottom: 15px;'>")
                for material in materials:
                    html_parts.append(f"<p style='margin: 5px 0;'>• {material}</p>")
                html_parts.append("</div>")
            
            # Additional Properties Section
            other_props = {}
            if hasattr(ifc_elem, "IsDefinedBy"):
                for rel in ifc_elem.IsDefinedBy:
                    if rel.is_a("IfcRelDefinesByProperties"):
                        props = rel.RelatingPropertyDefinition
                        if props.is_a("IfcPropertySet"):
                            for prop in props.HasProperties:
                                pname = prop.Name
                                val = getattr(getattr(prop, "NominalValue", None), "wrappedValue", None)
                                # Skip already displayed properties
                                if not any(keyword in pname.lower() for keyword in ['rt60', 'spl', 'acoustic', 'area', 'volume', 'height', 'width', 'length']):
                                    other_props[pname] = val
            
            # Add other properties from parsed data
            if global_id:
                parsed_data = self.find_element_data(global_id)
                for key, value in parsed_data.items():
                    if key.startswith('Property_'):
                        prop_name = key.replace('Property_', '')
                        # Skip already displayed properties
                        if not any(keyword in prop_name.lower() for keyword in ['rt60', 'spl', 'acoustic', 'area', 'volume', 'height', 'width', 'length']):
                            other_props[prop_name] = value
            
            if other_props:
                html_parts.append("<h3 style='color:#FF9800; border-bottom: 2px solid #FF9800; padding-bottom: 5px;'>📋 Additional Properties</h3>")
                html_parts.append("<div style='margin-bottom: 15px;'>")
                for prop_name, value in other_props.items():
                    formatted_value = self.format_property_value(value)
                    html_parts.append(f"<p style='margin: 5px 0;'><b>{prop_name}:</b> {formatted_value}</p>")
                html_parts.append("</div>")
            
            # Set the HTML content
            self.browser.setHtml("".join(html_parts))
            self.show()
            
        except Exception as e:
            error_html = f"""
                <div style="background: #ffebee; border: 1px solid #f44336; border-radius: 8px; padding: 15px; margin: 10px;">
                    <h3 style="color: #d32f2f; margin-top: 0;">❌ Error Displaying Element</h3>
                    <p style="color: #d32f2f;">{str(e)}</p>
                </div>
            """
            self.browser.setHtml(error_html)
            self.show()

    def get_apartment_legend_html(self):
        """Get HTML for apartment legend"""
        return """
        <div style="background: #2C2C2C; padding: 15px; border-radius: 10px; color: white;">
            <h3 style="margin-top: 0; color: #FF9500;">🏢 Apartment Type Legend</h3>
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
            <p style="margin-top: 15px; color: #FF9500; font-style: italic;">
                💡 Click on any element to see detailed information and run ML analysis
            </p>
        </div>
        """

    def set_apartment_overlay_active(self, active):
        """Set apartment overlay active state"""
        self.apartment_overlay_active = active

    def parse_apartment_info(self, space_name):
        """Parse apartment information from space name"""
        try:
            if not isinstance(space_name, str):
                return None
            
            # Example: "LVL2_3B_36" -> floor=2, type="3B", room=36
            parts = space_name.split("_")
            if len(parts) >= 2:
                floor_part = parts[0]
                type_part = parts[1]
                
                # Extract floor level
                floor = None
                if floor_part.startswith("LVL"):
                    try:
                        floor = int(floor_part.replace("LVL", ""))
                    except:
                        pass
                
                # Extract apartment type
                apt_type = None
                if type_part in ["1B", "2B", "3B"]:
                    apt_type = type_part
                
                # Extract room number
                room = None
                if len(parts) >= 3:
                    try:
                        room = int(parts[2])
                    except:
                        room = parts[2]  # Keep as string if not numeric
                
                return {
                    "floor": floor,
                    "type": apt_type,
                    "room": room
                }
            
            return None
        except Exception as e:
            print(f"Error parsing apartment info: {e}")
            return None

    def on_element_data_ready(self, element_data):
        """Handle when element data is ready for ML processing"""
        try:
            print(f"🎯 Element data ready for ML: {element_data.get('type', 'Unknown')}")
            # This signal can be connected to the main UI's ML pipeline
        except Exception as e:
            print(f"❌ Error handling element data ready: {e}")

    def run_ml_analysis(self):
        """Trigger ML analysis for the current selected element"""
        if not ML_AVAILABLE or geometry_ml_interface is None:
            self.browser.append("<p style='color:red;'>❌ ML interface not available</p>")
            return
            
        if not self.current_element_data:
            self.browser.append("<p style='color:red;'>❌ No element selected for ML analysis</p>")
            return
        
        try:
            print(f"🤖 Running ML analysis on element: {self.current_element_data.get('type', 'Unknown')}")
            
            # Extract data for ML processing
            extracted_data = geometry_ml_interface.extract_element_data(self.current_element_data)
            print(f"🔍 Extracted data for ML: {extracted_data}")
            
            # Run comfort prediction
            comfort_result = geometry_ml_interface.predict_comfort_for_element(extracted_data)
            print(f"🎯 Comfort prediction result: {comfort_result}")
            
            # Generate recommendations
            recommendations = geometry_ml_interface.get_ml_recommendations(extracted_data)
            print(f"💡 Recommendations: {recommendations}")
            
            # Display results
            self.display_ml_results(comfort_result, recommendations, extracted_data)
            
        except Exception as e:
            error_msg = f"❌ ML analysis failed: {str(e)}"
            print(error_msg)
            self.browser.append(f"<p style='color:red;'>{error_msg}</p>")
            import traceback
            traceback.print_exc()
    
    def display_ml_results(self, comfort_result, recommendations, extracted_data):
        """Display ML analysis results in the InfoDock"""
        html_parts = []
        html_parts.append("<h3 style='color:#FF9500;'>🤖 ML Analysis Results</h3>")
        
        if "error" in comfort_result:
            html_parts.append(f"<p style='color:red;'>❌ {comfort_result['error']}</p>")
        else:
            # Display comfort prediction
            html_parts.append("<h4 style='color:#4CAF50;'>🎯 Comfort Prediction</h4>")
            html_parts.append(f"<p><b>Comfort Score:</b> {comfort_result['comfort_score']}</p>")
            html_parts.append(f"<p><b>Confidence:</b> {comfort_result['confidence']:.1%}</p>")
            html_parts.append(f"<p><b>Element Type:</b> {comfort_result['element_type']}</p>")
            html_parts.append(f"<p><b>Apartment Type:</b> {comfort_result['apartment_type']}</p>")
            
            # Display key features used
            if 'features_used' in comfort_result:
                features = comfort_result['features_used']
                html_parts.append("<h4 style='color:#2196F3;'>📊 Key Features Used</h4>")
                if features.get('RT60(seconds)'):
                    html_parts.append(f"<p><b>RT60:</b> {features['RT60(seconds)']} seconds</p>")
                if features.get('SPL'):
                    html_parts.append(f"<p><b>SPL:</b> {features['SPL']} dBA</p>")
                if features.get('Height'):
                    html_parts.append(f"<p><b>Height:</b> {features['Height']} m</p>")
        
        # Display recommendations
        if recommendations:
            html_parts.append("<h4 style='color:#FF9800;'>💡 Recommendations</h4>")
            for rec in recommendations:
                html_parts.append(f"<p>• {rec}</p>")
        
        # Add separator
        html_parts.append("<hr style='border-color:#FF9500; margin:20px 0;'>")
        
        # Append to existing content
        current_html = self.browser.toHtml()
        new_html = current_html + "".join(html_parts)
        self.browser.setHtml(new_html)
    
    def extract_element_data_for_ml(self, ifc_elem):
        """Extract structured data from IFC element for ML processing"""
        try:
            data = {
                "type": ifc_elem.is_a(),
                "name": getattr(ifc_elem, "LongName", "") or getattr(ifc_elem, "Name", ""),
                "global_id": getattr(ifc_elem, 'GlobalId', ''),
                "description": getattr(ifc_elem, "Description", ""),
                "object_type": getattr(ifc_elem, "ObjectType", ""),
                "acoustic_properties": {},
                "dimensions": {},
                "location": {},
                "accessibility": {},
                "materials": [],
                "other_properties": {}
            }
            
            # Extract acoustic properties
            if hasattr(ifc_elem, "IsDefinedBy"):
                for rel in ifc_elem.IsDefinedBy:
                    if rel.is_a("IfcRelDefinesByProperties"):
                        props = rel.RelatingPropertyDefinition
                        if props.is_a("IfcPropertySet"):
                            for prop in props.HasProperties:
                                pname = prop.Name.lower()
                                val = getattr(getattr(prop, "NominalValue", None), "wrappedValue", None)
                                if "rt60" in pname:
                                    data["acoustic_properties"]["RT60"] = val
                                if "spl" in pname:
                                    data["acoustic_properties"]["SPL"] = val
            
            # Extract materials
            if hasattr(ifc_elem, "HasAssociations"):
                for assoc in ifc_elem.HasAssociations:
                    if assoc.is_a("IfcRelAssociatesMaterial"):
                        mat = assoc.RelatingMaterial
                        if hasattr(mat, "Name"):
                            data["materials"].append(mat.Name)
                        elif hasattr(mat, "ForLayerSet"):
                            layers = mat.ForLayerSet.MaterialLayers
                            for layer in layers:
                                if hasattr(layer.Material, "Name"):
                                    data["materials"].append(layer.Material.Name)
            
            # Get additional data from parsed files
            global_id = getattr(ifc_elem, 'GlobalId', '')
            if global_id:
                additional_data = self.find_element_data(global_id)
                if additional_data:
                    for key, value in additional_data.items():
                        if key.startswith('Property_'):
                            prop_name = key.replace('Property_', '')
                            if 'RT60' in prop_name or 'SPL' in prop_name:
                                data["acoustic_properties"][prop_name] = value
                            elif 'Area' in prop_name or 'Volume' in prop_name or 'GrossPlannedArea' in prop_name:
                                data["dimensions"][prop_name] = value
                            elif 'Location' in prop_name or 'X' in prop_name or 'Y' in prop_name or 'Z' in prop_name:
                                data["location"][prop_name] = value
                            elif 'Accessible' in prop_name or 'Public' in prop_name or 'External' in prop_name:
                                data["accessibility"][prop_name] = value
                            else:
                                data["other_properties"][prop_name] = value
            
            return data
            
        except Exception as e:
            print(f"Error extracting element data for ML: {e}")
            return None

    def clear_content(self):
        """Clear the InfoDock content"""
        if self.apartment_overlay_active:
            self.browser.setHtml(self.get_apartment_legend_html())
        else:
            self.browser.clear()
            self.hide()
        self.current_element_data = None
        self.ml_btn.setEnabled(False)

# Keep the original InfoDock for backward compatibility
class InfoDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("IFC Element Info", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.browser = QTextBrowser()
        self.browser.setStyleSheet("""
            background-color: #1A1A1A;
            color: snow;
            font-family: 'Segoe UI';
            font-weight: bold;
            padding: 8px;
        """)
        self.setWidget(self.browser)
        self.setFloating(False)
        self.hide()  # Start hidden

    def update_content(self, html):
        self.browser.setHtml(html)
        self.show() 