#!/usr/bin/env python3
"""
Apartment Color Overlay for IFC Viewer
Colors all elements within each apartment space based on apartment type (1B, 2B, 3B)
"""

from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
import ifcopenshell
import ifcopenshell.geom
from collections import defaultdict

def parse_apartment_info(space_name):
    """
    Parse apartment information from space name
    Example: "LVL1_3B_50" -> {"level": 1, "type": "3B", "number": 50}
    """
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
    """
    Get color for apartment type
    """
    color_map = {
        "1B": (0.2, 0.6, 1.0),    # Blue
        "2B": (0.2, 0.8, 0.2),    # Green  
        "3B": (1.0, 0.4, 0.2),    # Orange
    }
    return color_map.get(apartment_type, (0.7, 0.7, 0.7))  # Default gray

def find_apartment_spaces(ifc_file):
    """
    Find all apartment spaces and their properties
    """
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
    """
    Find all elements that belong to a specific space
    """
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
        # Look for elements that reference this space
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
    """
    Create apartment color overlay for all elements
    """
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
    """
    Apply apartment colors to the viewer shapes
    """
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

def create_apartment_legend():
    """
    Create a legend showing apartment type colors
    """
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

def get_apartment_statistics(viewer, ifc_file_path):
    """
    Get statistics about apartments and their elements
    """
    try:
        ifc_file = ifcopenshell.open(ifc_file_path)
        apartment_spaces = find_apartment_spaces(ifc_file)
        
        stats = {
            "total_apartments": len(apartment_spaces),
            "apartments_by_type": defaultdict(int),
            "apartments_by_level": defaultdict(int),
            "elements_by_apartment": defaultdict(list)
        }
        
        for space_global_id, space_data in apartment_spaces.items():
            apt_info = space_data["info"]
            apt_type = apt_info["type"]
            level = apt_info["level"]
            
            stats["apartments_by_type"][apt_type] += 1
            stats["apartments_by_level"][level] += 1
            
            # Count elements in this apartment
            elements = find_elements_in_space(ifc_file, space_global_id)
            stats["elements_by_apartment"][apt_type].extend(elements)
        
        return stats
        
    except Exception as e:
        print(f"❌ Error getting apartment statistics: {e}")
        return None

# Integration functions for your existing viewer
def add_apartment_overlay_button(viewer, main_window):
    """
    Add apartment overlay button to your main window
    """
    from PyQt6.QtWidgets import QPushButton
    
    overlay_btn = QPushButton("Apply Apartment Overlay")
    overlay_btn.setStyleSheet(f"background: {COLOR_INPUT_BG}; color: {COLOR_INPUT_TEXT}; font-weight: bold;")
    
    def on_overlay_clicked():
        if hasattr(viewer, 'last_loaded_model_path') and viewer.last_loaded_model_path:
            stats = create_apartment_overlay(viewer, viewer.last_loaded_model_path)
            if stats:
                # Update info dock with statistics
                if hasattr(viewer, 'info_dock') and viewer.info_dock:
                    html_content = f"""
                    <h3>Apartment Overlay Applied</h3>
                    <p><strong>Total Apartments:</strong> {stats['total_apartments']}</p>
                    <p><strong>Total Elements:</strong> {stats['total_elements']}</p>
                    <h4>Elements by Apartment Type:</h4>
                    <ul>
                    """
                    for apt_type, count in stats['apartment_counts'].items():
                        html_content += f"<li><strong>{apt_type}:</strong> {count} elements</li>"
                    html_content += "</ul>"
                    html_content += create_apartment_legend()
                    
                    viewer.info_dock.update_content(html_content)
                
                QMessageBox.information(main_window, "Success", 
                    f"Apartment overlay applied!\n"
                    f"Total apartments: {stats['total_apartments']}\n"
                    f"Total elements: {stats['total_elements']}")
            else:
                QMessageBox.warning(main_window, "Error", "Could not apply apartment overlay")
        else:
            QMessageBox.warning(main_window, "Error", "No IFC model loaded")
    
    overlay_btn.clicked.connect(on_overlay_clicked)
    return overlay_btn

def reset_apartment_colors(viewer):
    """
    Reset colors to original element type colors
    """
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

# Example usage
if __name__ == "__main__":
    print("""
🏢 Apartment Color Overlay for IFC Viewer

To use this in your ui8.py:

1. Add the import:
   from scripts.apartment_color_overlay import *

2. Add the overlay button to your main window:
   overlay_btn = add_apartment_overlay_button(self.viewer, self)
   # Add the button to your layout

3. The overlay will:
   - Parse apartment info from space names (LVL1_3B_50)
   - Find all elements within each apartment
   - Apply color coding: 1B=Blue, 2B=Green, 3B=Orange
   - Update the viewer with apartment-based colors

4. Optional: Add reset button:
   reset_btn = QPushButton("Reset Colors")
   reset_btn.clicked.connect(lambda: reset_apartment_colors(self.viewer))
    """) 