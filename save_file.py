import os
from pathlib import Path

def save_file(file_name, df):
    file_folder = "Files"
    output_path = f"{file_folder}//{file_name}"
    if os.path.exists(output_path):
        os.remove(output_path)
        
    filepath = Path(output_path)
    df.to_csv(filepath)
    print(f"Output saved to file {filepath}")