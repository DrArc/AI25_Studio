#!/usr/bin/env python3
"""
Render Quality Enhancer for PyQt6 + OpenCascade IFC Viewer
Add these functions to your existing ui8.py to improve render quality
"""

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
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB

def enhance_viewer_rendering(viewer):
    """
    Apply enhanced rendering settings to your existing viewer
    
    Usage: Add this to your MyViewer.__init__() method:
        enhance_viewer_rendering(self)
    """
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
    """
    Improve the tessellation quality of a shape
    
    Usage: Call this before displaying shapes:
        improved_shape = improve_shape_tessellation(original_shape, 0.1)
    """
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
    """
    Apply enhanced materials based on element type
    
    Usage: Call this after creating AIS shapes:
        apply_enhanced_materials(self, ais_shape, element_type)
    """
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
    """
    Create enhanced IFC geometry settings
    
    Usage: Replace your existing settings with this:
        settings = create_enhanced_ifc_settings()
    """
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_PYTHON_OPENCASCADE, True)
    
    # Enhanced tessellation settings
    settings.set(settings.INCLUDE_CURVES, True)
    settings.set(settings.SEW_SHELLS, True)
    settings.set(settings.USE_WORLD_COORDS, True)
    
    return settings

def enhanced_load_ifc_file(viewer, path, tessellation_quality=0.1):
    """
    Enhanced IFC loading function with better quality
    
    Usage: Replace your existing load_ifc_file method with this enhanced version
    """
    from OCC.Core.Graphic3d import Graphic3d_NameOfMaterial

    # Step 1: Clear previous shapes
    viewer._display.EraseAll()
    viewer._display.Context.RemoveAll(True)
    viewer.shape_to_ifc.clear()

    # Step 2: Setup enhanced rendering
    enhance_viewer_rendering(viewer)

    # Step 3: Load new model with enhanced settings
    settings = create_enhanced_ifc_settings()

    try:
        ifc = ifcopenshell.open(path)
        products = ifc.by_type("IfcProduct")
        
        print(f"📊 Loading {len(products)} IFC products with enhanced quality...")
        
        for i, p in enumerate(products):
            if hasattr(p, "Representation") and p.Representation:
                try:
                    # Create shape with enhanced settings
                    shape_result = ifcopenshell.geom.create_shape(settings, p)
                    shape = shape_result.geometry
                    
                    # Improve shape quality
                    shape = improve_shape_tessellation(shape, tessellation_quality)
                    
                    # Display shape with enhanced quality
                    ais_result = viewer._display.DisplayShape(shape, update=False)
                    if not isinstance(ais_result, list):
                        ais_result = [ais_result]
                        
                    for ais_obj in ais_result:
                        viewer.shape_to_ifc[ais_obj] = p
                        
                        # Apply enhanced materials
                        element_type = p.is_a()
                        apply_enhanced_materials(viewer, ais_obj, element_type)
                        
                        # Apply colors based on element type
                        color = viewer.color_map.get(element_type, viewer.default_color)
                        color_obj = Quantity_Color(color[0], color[1], color[2], Quantity_TOC_RGB)
                        viewer._display.Context.SetColor(ais_obj, color_obj, False)
                    
                    # Progress update
                    if i % 10 == 0:
                        print(f"📈 Processed {i}/{len(products)} elements...")
                        
                except Exception as e:
                    print(f"⚠️ Could not render shape for: {p}. Error: {e}")
        
        # Final update and fit
        viewer._display.Context.UpdateCurrentViewer()
        viewer._display.FitAll()
        viewer.last_loaded_model_path = path
        
        print(f"✅ Successfully loaded IFC model with enhanced quality!")
        print(f"📊 Total elements: {len(viewer.shape_to_ifc)}")
        
    except Exception as e:
        print(f"❌ Failed to load IFC: {e}")

def add_quality_controls_to_main_window(main_window):
    """
    Add quality control widgets to your main window
    
    Usage: Call this in your main window __init__ method:
        add_quality_controls_to_main_window(self)
    """
    from PyQt6.QtWidgets import QGroupBox, QSlider, QCheckBox, QLabel, QVBoxLayout, QHBoxLayout
    
    # Create quality control panel
    quality_group = QGroupBox("Render Quality Controls")
    quality_layout = QVBoxLayout(quality_group)
    
    # Quality slider
    quality_label = QLabel("Tessellation Quality:")
    quality_slider = QSlider(Qt.Orientation.Horizontal)
    quality_slider.setRange(1, 10)  # 1 = best quality, 10 = worst
    quality_slider.setValue(3)  # Default to good quality
    quality_slider.setToolTip("Lower values = better quality but slower rendering")
    
    def on_quality_changed(value):
        quality = 0.1 * value  # Convert to tessellation quality
        print(f"🎨 Tessellation quality set to: {quality}")
        # You can store this value and use it in your load function
    
    quality_slider.valueChanged.connect(on_quality_changed)
    
    # Anti-aliasing checkbox
    aa_checkbox = QCheckBox("Enable Anti-Aliasing")
    aa_checkbox.setChecked(True)
    
    def on_aa_toggled(checked):
        try:
            main_window.viewer._display.Context.SetAntialiasing(checked)
            print(f"🔄 Anti-aliasing {'enabled' if checked else 'disabled'}")
        except Exception as e:
            print(f"⚠️ Could not toggle anti-aliasing: {e}")
    
    aa_checkbox.toggled.connect(on_aa_toggled)
    
    # Shadows checkbox
    shadows_checkbox = QCheckBox("Enable Shadows")
    shadows_checkbox.setChecked(True)
    
    def on_shadows_toggled(checked):
        try:
            main_window.viewer._display.Context.SetShadows(checked)
            print(f"🌑 Shadows {'enabled' if checked else 'disabled'}")
        except Exception as e:
            print(f"⚠️ Could not toggle shadows: {e}")
    
    shadows_checkbox.toggled.connect(on_shadows_toggled)
    
    # Add widgets to layout
    quality_layout.addWidget(quality_label)
    quality_layout.addWidget(quality_slider)
    quality_layout.addWidget(aa_checkbox)
    quality_layout.addWidget(shadows_checkbox)
    
    # Add to your main window layout (adjust as needed)
    # main_window.left_panel.layout().addWidget(quality_group)
    
    return quality_group

# Example usage instructions
if __name__ == "__main__":
    print("""
🎨 Render Quality Enhancer for PyQt6 + OpenCascade
    
To use these enhancements in your ui8.py:

1. Add the imports at the top of your file:
   from scripts.render_quality_enhancer import *

2. In your MyViewer.__init__() method, add:
   enhance_viewer_rendering(self)

3. Replace your load_ifc_file method with:
   def load_ifc_file(self, path):
       enhanced_load_ifc_file(self, path, tessellation_quality=0.1)

4. In your main window, add quality controls:
   quality_panel = add_quality_controls_to_main_window(self)
   # Add the panel to your layout

5. For even better quality, you can also:
   - Set tessellation_quality to 0.05 for ultra quality
   - Enable shadows and anti-aliasing
   - Use the enhanced material mapping

These changes will significantly improve your render quality!
    """) 