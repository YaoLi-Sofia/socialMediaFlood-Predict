"""
Author: Joel
FilePath: lstm/weather/next_file_name.py
Date: 2025-03-17 22:48:02
LastEditTime: 2025-03-18 15:31:51
Description: 
"""
import os


def get_next_file_name(directory, base_name, extension):
    if not os.path.exists(directory):
        os.makedirs(directory)
    existing_files = [
        filename
        for filename in os.listdir(directory)
        if filename.endswith(extension) and filename.startswith(base_name)
    ]
    if existing_files:
        max_index = 0
        for filename in existing_files:
            index = int(filename.replace(base_name, '').rstrip(extension).rstrip(''))
            max_index = max(max_index, index)
        next_index = max_index + 1
        return f"{base_name}{str(next_index)}{extension}"
    else:
        return f"{base_name}1{extension}"
