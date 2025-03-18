"""
Author: Joel
FilePath: lstm/lstm_sinewave/next_file_name.py
Date: 2025-03-16 12:56:36
LastEditTime: 2025-03-16 13:13:41
Description: 
"""
import os


# base_name = 'predictions'
# directory = 'stock_result'
# extension = '.csv'


def get_next_predictions_name(base_name, directory, extension):
    if not os.path.exists(directory):
        os.makedirs(directory)
    existing_files = [
        file_name for file_name in os.listdir(directory)
        if file_name.endswith(extension) and file_name.startswith(base_name)
    ]
    max_index = 0
    for file in existing_files:
        try:
            prefix_name, ext = os.path.splitext(file)
            index = int(prefix_name.replace(base_name, '').strip(''))
            max_index = max(index, max_index)
        except ValueError:
            continue
    next_index = max_index + 1
    next_name = f"{base_name}{str(next_index)}{extension}"
    return next_name
