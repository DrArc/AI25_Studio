# reference_data.py

import re

DAY_RANGES = {
    "HD-Urban-V0": (35.0, 60.0),
    "HD-Urban-V1": (15.0, 60.0),
    "MD-Urban-V2": (15.0, 40.0),
    "LD-Urban-V3": (15.0, 40.0),
    "Ind-Zone-V0": (15.0, 50.0),
    "Roadside-V1": (10.0, 55.0),
    "Roadside-V2": (15.0, 50.0),
    "Roadside-V3": (15.0, 55.0),
    "GreenEdge-V3": (10.0, 35.0),
}

NIGHT_RANGES = {
    "HD-Urban-V0": (10.0, 30.0),
    "HD-Urban-V1": (15.0, 30.0),
    "MD-Urban-V2": (12.0, 28.0),
    "LD-Urban-V3": (18.0, 35.0),
    "Ind-Zone-V0": (40.0, 45.0),
    "Roadside-V1": (35.0, 50.0),
    "Roadside-V2": (30.0, 45.0),
    "Roadside-V3": (30.0, 45.0),

    "GreenEdge-V3": (20.0, 35.0), 
}


material_directory = {
    'window': [
        (0.10, "Single Pane Glass"),
        (0.18, "Double Pane Glass"),
        (0.20, "Laminated Glass"),
        (0.05, "Wired Glass"),
        (0.06, "Frosted Glass"),
        (0.03, "Insulated Glazing Unit"),
        (0.01, "Glass Block"),
        (0.01, "Glazed Ceramic Tile"),
        (0.04, "Large Pane Glass"),
        (0.03, "Small Pane Glass"),
    ],
    'wall': [
        (0.01, "Painted Brick"),
        (0.03, "Unpainted Brick"),
        (0.36, "Concrete Block (Coarse)"),
        (0.46, "Concrete Block (Painted)"),
        (0.05, "Gypsum Board"),
        (0.02, "Plaster on Masonry"),
        (0.04, "Plaster with Wallpaper Backing"),
        (0.15, "Wood Paneling"),
        (0.50, "Acoustic Plaster"),
        (0.65, "Fiberglass Board"),
    ],
    'floor': [
        (0.01, "Marble"),
        (0.01, "Terrazzo"),
        (0.03, "Vinyl Tile"),
        (0.07, "Wood Parquet"),
        (0.10, "Wood Flooring on Joists"),
        (0.25, "Thin Carpet on Concrete"),
        (0.30, "Thin Carpet on Wood"),
        (0.50, "Medium Pile Carpet"),
        (0.50, "Thick Pile Carpet"),
        (0.15, "Cork Floor Tiles"),
    ],
    'door': [
        (0.06, "Solid Wood Door"),
        (0.15, "Hollow-Core Wood Door"),
        (0.44, "Acoustic Door"),
        (0.49, "Steel Door with Acoustic Treatment"),
        (0.10, "Plywood Door"),
        (0.04, "Single Pane Glass Door"),
        (0.03, "Double Pane Glass Door"),
        (0.20, "Laminated Glass Door"),
        (0.05, "Sliding Wood Door"),
        (0.35, "Fire-Rated Door with Mineral Core"),
    ]
}

# Comfort index thresholds
LAeq_target = 30.0
LAeq_min = 20.0
LAeq_max = 85.0
RT60_target = 0.4
RT60_max_dev = 0.9
RT60_min = 0.2

def detect_apartment_type(name):
    # Accepts 1B, 2B, 3B, 1BED, 2BED, 3BED, with or without underscores, case-insensitive
    match = re.search(r'([123])[_ ]?B(ED)?', name.upper())
    if match:
        return f"{match.group(1)}B"
    return None