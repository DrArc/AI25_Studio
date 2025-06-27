import ifcopenshell
import os
import json
import pandas as pd
from pathlib import Path

input_dir = Path("models")
output_dir = input_dir / "parse"
output_dir.mkdir(parents=True, exist_ok=True)

def extract_ifcspace_data(ifc_file_path):
    try:
        ifc_file = ifcopenshell.open(ifc_file_path)
    except Exception as e:
        return {"error": f"Failed to open file {ifc_file_path}: {e}"}
    
    data = []
    for space in ifc_file.by_type("IfcSpace"):
        props = {
            "GlobalId": getattr(space, "GlobalId", ""),
            "Name": getattr(space, "Name", ""),
            "Description": getattr(space, "Description", ""),
            "ObjectType": getattr(space, "ObjectType", ""),
            "IfcEntity": space.is_a()
        }

        if hasattr(space, "IsDefinedBy"):
            for rel in space.IsDefinedBy:
                if rel.is_a("IfcRelDefinesByProperties"):
                    prop_def = rel.RelatingPropertyDefinition
                    if prop_def.is_a("IfcPropertySet"):
                        for prop in prop_def.HasProperties:
                            pname = prop.Name
                            val = getattr(getattr(prop, "NominalValue", None), "wrappedValue", None)
                            if pname and val is not None:
                                props[f"Property_{pname}"] = val
        data.append(props)
    
    return data

# Process all IFC files in models/
for ifc_path in input_dir.glob("*.ifc"):
    base_name = ifc_path.stem
    extracted_data = extract_ifcspace_data(ifc_path)
    
    if isinstance(extracted_data, dict) and "error" in extracted_data:
        print(extracted_data["error"])
        continue

    # Save both JSON and CSV
    (output_dir / f"{base_name}.json").write_text(json.dumps(extracted_data, indent=4))
    pd.DataFrame(extracted_data).to_csv(output_dir / f"{base_name}.csv", index=False)
    print(f"✅ Extracted {len(extracted_data)} spaces from {ifc_path.name}")
