# Core Module - ML Integration Documentation

## Overview
This module provides comprehensive ML integration for IFC element analysis and acoustic comfort prediction. The system connects IFC geometry data from the 3D viewer to ML models for real-time acoustic analysis.

## Key Components

### 1. EnhancedInfoDock (`fix_occ_import_error.py`)
- **Purpose**: Displays detailed IFC element information and integrates with ML pipeline
- **Features**:
  - Extracts comprehensive element data from IFC files
  - Loads parsed CSV/JSON data for enhanced information
  - Provides ML analysis button for selected elements
  - Shows apartment legend when overlay is active
  - Emits signals when element data is ready for ML processing

### 2. Geometry ML Interface (`geometry_ml_interface.py`)
- **Purpose**: Connects IFC element data to ML models
- **Features**:
  - Loads trained comfort and acoustic models
  - Extracts structured data from IFC elements
  - Prepares features for ML prediction
  - Generates acoustic comfort predictions
  - Provides ML-based recommendations

### 3. Acoustic Pipeline (`acoustic_pipeline.py`)
- **Purpose**: Orchestrates the complete ML analysis workflow
- **Features**:
  - Combines SQL database lookups with ML predictions
  - Handles both standard and enhanced (geometry-based) pipelines
  - Integrates with LLM for natural language responses

### 4. LLM Integration (`llm_calls.py`, `sql_calls.py`)
- **Purpose**: Provides natural language processing and database queries
- **Features**:
  - Extracts structured parameters from user questions
  - Queries SQL database for existing comfort data
  - Generates comprehensive summaries with recommendations

## Data Flow

### Element Selection Flow:
1. **User clicks on IFC element** in 3D viewer
2. **EnhancedInfoDock** extracts comprehensive element data
3. **Element data** is automatically passed to ML pipeline
4. **ML analysis** runs for relevant element types (IfcSpace, IfcWall, IfcSlab, IfcColumn, IfcBeam)
5. **Results** are displayed in both InfoDock and chat panel

### ML Processing Flow:
1. **Extract element data** from IFC properties and parsed files
2. **Prepare features** using `infer_features()` function
3. **Override with actual measurements** (RT60, SPL, area) if available
4. **Run comfort prediction** using trained ML model
5. **Generate recommendations** based on element properties
6. **Display results** with confidence scores

## Usage Instructions

### For Users:
1. **Load IFC file** using "3D File Upload" button
2. **Click on any element** in the 3D viewer
3. **View detailed information** in the InfoDock panel
4. **ML analysis runs automatically** for relevant elements
5. **Results appear** in the chat panel with recommendations

### For Developers:
1. **Element selection** triggers `check_occ_selection()` in main UI
2. **InfoDock** extracts data via `extract_element_data_for_ml()`
3. **ML interface** processes data via `extract_element_data()` and `predict_comfort_for_element()`
4. **Results** are displayed via `run_ml_analysis_for_element()`

## Supported Element Types

### Automatic ML Analysis:
- `IfcSpace` - Apartment spaces with acoustic properties
- `IfcWall` - Wall elements with material properties
- `IfcSlab` - Floor/ceiling elements
- `IfcColumn` - Structural columns
- `IfcBeam` - Structural beams

### Manual ML Analysis:
- Any element can be analyzed using the "🤖 Run ML Analysis" button in InfoDock

## Data Sources

### IFC Properties:
- Basic element information (type, name, description)
- Acoustic properties (RT60, SPL)
- Material associations
- Geometric properties

### Parsed Files:
- Enhanced properties from CSV/JSON files in `models/parse/` directory
- Additional acoustic measurements
- Detailed material specifications
- Spatial relationships

### ML Model Features:
- Zone and apartment type information
- Floor height and surface area
- Absorption coefficients
- Sound source information
- Barrier properties

## Troubleshooting

### Common Issues:

1. **"ML interface not available"**
   - Check that `geometry_ml_interface.py` is properly imported
   - Verify ML model files exist in `model/` directory

2. **"No element data available"**
   - Ensure IFC file is loaded
   - Check that element has geometric representation
   - Verify parsed data files exist

3. **"ML Prediction Error"**
   - Check feature preparation in `prepare_features_for_comfort_model()`
   - Verify ML model compatibility
   - Review console output for detailed error messages

### Debug Commands:
- Use "Debug ML Integration" button to test ML pipeline
- Check console output for detailed processing information
- Review InfoDock content for element data extraction

## File Structure
```
scripts/core/
├── fix_occ_import_error.py      # Enhanced InfoDock with ML integration
├── geometry_ml_interface.py     # ML interface for IFC elements
├── acoustic_pipeline.py         # Complete ML analysis pipeline
├── llm_calls.py                 # LLM integration for natural language
├── sql_calls.py                 # Database queries and fallback
├── sql_query_handler.py         # SQL-specific query handling
├── llm_acoustic_query_handler.py # Acoustic-specific LLM queries
└── README.md                    # This documentation
```

## Integration Points

### Main UI (`ui10.py`):
- Connects InfoDock to ML pipeline
- Handles element selection events
- Displays ML results in chat panel
- Manages apartment overlay integration

### Utils (`utils/infer_from_inputs.py`):
- Provides feature inference for ML models
- Handles data standardization
- Manages fallback calculations

### Models (`model/`):
- Contains trained ML models for comfort prediction
- Stores acoustic analysis models
- Provides model loading functionality

## Future Enhancements

1. **Real-time acoustic simulation** based on element properties
2. **Material optimization** recommendations
3. **Compliance checking** against acoustic standards
4. **Batch analysis** for multiple elements
5. **Export functionality** for analysis reports 