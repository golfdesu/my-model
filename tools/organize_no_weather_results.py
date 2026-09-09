#!/usr/bin/env python
# coding: utf-8

"""
organize_no_weather_results.py
Moves benchmark artifacts (*_results.json, *_best.pt, *_predictions.npz)
from the root directory into outputs/acn_caltech_no_weather/<model_name>/.
"""

import os
import glob
import shutil

TARGET_DIR = os.path.join("outputs", "acn_caltech_no_weather")

def organize():
    os.makedirs(TARGET_DIR, exist_ok=True)
    extensions = ["_results.json", "_best.pt", "_predictions.npz"]
    
    moved_count = 0
    for ext in extensions:
        files = glob.glob(f"*{ext}")
        for file_path in files:
            model_name = file_path.replace(ext, "")
            model_folder = os.path.join(TARGET_DIR, model_name)
            os.makedirs(model_folder, exist_ok=True)
            
            dest_path = os.path.join(model_folder, file_path)
            shutil.move(file_path, dest_path)
            print(f"Moved: {file_path} -> {dest_path}")
            moved_count += 1
            
    if moved_count == 0:
        print("No output files found in the root directory to organize.")
    else:
        print(f"\nSuccessfully organized {moved_count} files into {TARGET_DIR}/")

if __name__ == "__main__":
    organize()
