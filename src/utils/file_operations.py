"""
File operation utilities for the CircuitGuard-PCB project
"""

import os
import yaml
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Any, filepath: str):
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def create_directory(path: str):
    """
    Create directory if it doesn't exist
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)


def get_all_groups(pcb_data_path: str) -> List[str]:
    """
    Get all group folders from PCBData directory
    
    Args:
        pcb_data_path: Path to PCBData folder
        
    Returns:
        List of group folder names
    """
    groups = []
    for item in os.listdir(pcb_data_path):
        item_path = os.path.join(pcb_data_path, item)
        if os.path.isdir(item_path) and item.startswith('group'):
            groups.append(item)
    return sorted(groups)


def copy_file(src: str, dst: str):
    """
    Copy file from source to destination
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)