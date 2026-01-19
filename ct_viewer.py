#!/usr/bin/env python3
"""
CT Viewer - Universal CT/MRI Image Viewer
Supports DICOM files and JPG image sequences
Web-based 3D visualization with cross-sectional MPR views

Features:
- Directory picker with recent history
- Auto-detection of image format (DICOM or JPG)
- 3D reconstruction with noise reduction
- Interactive cross-sectional navigation
"""

import os
import io
import re
import json
import base64
import numpy as np
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request, redirect, url_for
from skimage import measure, morphology
from scipy import ndimage
from PIL import Image

# Optional DICOM support
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

app = Flask(__name__)

# Configuration
CONFIG_FILE = Path(__file__).parent / "recent_directories.json"
PORT = 7002

# Global state
current_volume = None
current_spacing = None
current_shape = None
current_mesh = None
current_directory = None
current_scan_type = None  # 'dental', 'thorax', 'head', 'body', or 'unknown'
current_dicom_metadata = {}

# Scan type presets: (threshold, window_center, window_width, keep_n_largest, mesh_color)
SCAN_PRESETS = {
    'dental': {
        'threshold': 600,
        'window_center': 400,
        'window_width': 2000,
        'keep_n_largest': 3,
        'mesh_color': 0xf5f5dc,  # Bone/beige
        'downsample_threshold': 400**3,
        'use_morphological_closing': True,
    },
    'bone': {
        'threshold': 300,
        'window_center': 400,
        'window_width': 2000,
        'keep_n_largest': 5,
        'mesh_color': 0xf5f5dc,
        'downsample_threshold': 300**3,
        'use_morphological_closing': True,
    },
    'soft_tissue': {
        'threshold': 50,
        'window_center': 40,
        'window_width': 400,
        'keep_n_largest': 3,
        'mesh_color': 0xffcccc,
        'downsample_threshold': 300**3,
        'use_morphological_closing': False,
    },
    'lung': {
        'threshold': -400,
        'window_center': -600,
        'window_width': 1500,
        'keep_n_largest': 3,
        'mesh_color': 0xccccff,
        'downsample_threshold': 300**3,
        'use_morphological_closing': False,
    },
    'auto': {
        'threshold': None,  # Will be auto-detected
        'window_center': None,
        'window_width': None,
        'keep_n_largest': 3,
        'mesh_color': 0xcccccc,
        'downsample_threshold': 300**3,
        'use_morphological_closing': False,
    }
}

def load_recent_directories():
    """Load list of recently used directories, filtering out non-existent ones."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                directories = json.load(f)
            # Filter out directories that no longer exist
            valid_directories = [d for d in directories if os.path.isdir(d)]
            # Save cleaned list if any were removed
            if len(valid_directories) != len(directories):
                save_recent_directories(valid_directories)
            return valid_directories
        except:
            pass
    return []

def save_recent_directories(directories):
    """Save list of recently used directories."""
    # Keep only the 20 most recent
    directories = directories[:20]
    with open(CONFIG_FILE, 'w') as f:
        json.dump(directories, f, indent=2)

def add_recent_directory(path):
    """Add a directory to recent list."""
    directories = load_recent_directories()
    # Remove if already exists (will re-add at top)
    if path in directories:
        directories.remove(path)
    # Add to beginning
    directories.insert(0, path)
    save_recent_directories(directories)

def get_folder_info(path):
    """Get list of subfolders and count of DCM/image files in a directory."""
    folders = []
    dcm_count = 0
    try:
        for item in sorted(os.listdir(path)):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                try:
                    # Count DICOM and image files
                    files = os.listdir(item_path)
                    dcm_files = [f for f in files if f.lower().endswith('.dcm')]
                    jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg'))]
                    png_files = [f for f in files if f.lower().endswith('.png')]
                    total_images = len(dcm_files) + len(jpg_files) + len(png_files)
                    folders.append({
                        'name': item,
                        'path': item_path,
                        'has_dcm': total_images > 0,
                        'dcm_count': total_images
                    })
                except PermissionError:
                    folders.append({'name': item, 'path': item_path, 'has_dcm': False, 'dcm_count': 0})
            elif item.lower().endswith(('.dcm', '.jpg', '.jpeg', '.png')):
                dcm_count += 1
    except PermissionError:
        pass
    return folders, dcm_count

def detect_image_type(directory):
    """Detect whether directory contains DICOM or JPG files."""
    files = os.listdir(directory)

    dcm_count = sum(1 for f in files if f.lower().endswith('.dcm'))
    jpg_count = sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg')))
    png_count = sum(1 for f in files if f.lower().endswith('.png'))

    if dcm_count > 0:
        return 'dicom', dcm_count
    elif jpg_count > 0:
        return 'jpg', jpg_count
    elif png_count > 0:
        return 'png', png_count
    else:
        return None, 0

def detect_scan_type(directory, volume_shape, dicom_metadata=None):
    """Auto-detect scan type based on metadata, path, and volume characteristics."""
    scan_type = 'auto'
    confidence = 0

    path_lower = directory.lower()

    # Check path for hints
    if 'dental' in path_lower or 'cbct' in path_lower or 'tooth' in path_lower or 'jaw' in path_lower:
        scan_type = 'dental'
        confidence = 0.8
    elif 'thorax' in path_lower or 'chest' in path_lower or 'lung' in path_lower:
        scan_type = 'lung'
        confidence = 0.7
    elif 'head' in path_lower or 'brain' in path_lower or 'cranial' in path_lower:
        scan_type = 'bone'
        confidence = 0.6

    # Check DICOM metadata if available
    if dicom_metadata:
        body_part = dicom_metadata.get('BodyPartExamined', '').lower()
        study_desc = dicom_metadata.get('StudyDescription', '').lower()
        series_desc = dicom_metadata.get('SeriesDescription', '').lower()
        modality = dicom_metadata.get('Modality', '').upper()

        all_desc = f"{body_part} {study_desc} {series_desc}"

        if any(x in all_desc for x in ['dental', 'tooth', 'jaw', 'mandible', 'maxilla', 'cbct']):
            scan_type = 'dental'
            confidence = 0.95
        elif any(x in all_desc for x in ['thorax', 'chest', 'lung', 'pulmonary']):
            scan_type = 'lung'
            confidence = 0.9
        elif body_part in ['head', 'brain', 'skull']:
            scan_type = 'bone'
            confidence = 0.8

        # CBCT modality often indicates dental
        if modality == 'CT' and 'cbct' in all_desc:
            scan_type = 'dental'
            confidence = 0.95

    # Check volume aspect ratio - dental CBCT tends to be more cubic
    if volume_shape and confidence < 0.7:
        z, y, x = volume_shape
        aspect_ratio = z / max(x, y) if max(x, y) > 0 else 1

        # Dental CBCT typically has aspect ratio close to 1 (cubic-ish)
        # Body CT is typically elongated (many more slices than width)
        if 0.5 < aspect_ratio < 1.5 and max(x, y) < 600:
            # Could be dental - small cubic volume
            if scan_type == 'auto':
                scan_type = 'dental'
                confidence = 0.5
        elif aspect_ratio > 2:
            # Elongated - likely full body or thorax
            if scan_type == 'auto':
                scan_type = 'auto'  # Keep as auto for body scans

    print(f"Detected scan type: {scan_type} (confidence: {confidence:.0%})")
    return scan_type

def natural_sort_key(s):
    """Sort strings with numbers naturally (1, 2, 10 instead of 1, 10, 2)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]

def load_image_series(directory):
    """Load JPG or PNG image sequence as 3D volume."""
    print(f"Loading image series from: {directory}")

    # Find all JPG and PNG files
    files = sorted([f for f in os.listdir(directory)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
                   key=natural_sort_key)

    if not files:
        raise ValueError("No image files found (JPG or PNG)")

    print(f"Found {len(files)} image files")

    # Load first image to get dimensions
    first_img = Image.open(os.path.join(directory, files[0]))
    if first_img.mode != 'L':
        first_img = first_img.convert('L')  # Convert to grayscale
    width, height = first_img.size

    print(f"Image dimensions: {width} x {height}")

    # Create volume
    num_slices = len(files)
    volume = np.zeros((num_slices, height, width), dtype=np.int16)

    for i, filename in enumerate(files):
        img = Image.open(os.path.join(directory, filename))
        if img.mode != 'L':
            img = img.convert('L')
        # Scale 0-255 to approximate HU-like range for visualization
        volume[i, :, :] = np.array(img, dtype=np.int16) * 4 - 500

        if (i + 1) % 50 == 0:
            print(f"Loaded {i + 1}/{num_slices} slices...")

    # Assume isotropic spacing for JPG (can be adjusted)
    spacing = (1.0, 1.0, 1.0)

    print(f"Volume loaded. Shape: {volume.shape}, Range: [{volume.min()}, {volume.max()}]")

    return volume, spacing

def load_dicom_series(directory):
    """Load DICOM series as 3D volume. Returns (volume, spacing, metadata)."""
    global current_dicom_metadata

    if not DICOM_AVAILABLE:
        raise ImportError("pydicom not available")

    print(f"Loading DICOM series from: {directory}")

    dicom_files = []
    for f in os.listdir(directory):
        if f.lower().endswith('.dcm'):
            filepath = os.path.join(directory, f)
            try:
                ds = pydicom.dcmread(filepath)
                if hasattr(ds, 'pixel_array'):
                    dicom_files.append((filepath, ds))
            except:
                pass

    if not dicom_files:
        raise ValueError("No valid DICOM files found")

    print(f"Found {len(dicom_files)} DICOM files")

    # Sort by slice location or instance number
    def get_sort_key(item):
        ds = item[1]
        if hasattr(ds, 'SliceLocation'):
            return float(ds.SliceLocation)
        elif hasattr(ds, 'InstanceNumber'):
            return int(ds.InstanceNumber)
        else:
            match = re.search(r'(\d+)', os.path.basename(item[0]))
            return int(match.group(1)) if match else 0

    dicom_files.sort(key=get_sort_key)

    first_ds = dicom_files[0][1]
    rows = first_ds.Rows
    cols = first_ds.Columns
    num_slices = len(dicom_files)

    # Extract metadata for scan type detection
    metadata = {
        'BodyPartExamined': getattr(first_ds, 'BodyPartExamined', ''),
        'StudyDescription': getattr(first_ds, 'StudyDescription', ''),
        'SeriesDescription': getattr(first_ds, 'SeriesDescription', ''),
        'Modality': getattr(first_ds, 'Modality', ''),
        'Manufacturer': getattr(first_ds, 'Manufacturer', ''),
        'InstitutionName': getattr(first_ds, 'InstitutionName', ''),
    }
    current_dicom_metadata = metadata
    print(f"DICOM Metadata: BodyPart={metadata['BodyPartExamined']}, Study={metadata['StudyDescription']}")

    print(f"Volume dimensions: {cols} x {rows} x {num_slices}")

    # Get spacing
    pixel_spacing = [1.0, 1.0]
    if hasattr(first_ds, 'PixelSpacing'):
        pixel_spacing = [float(first_ds.PixelSpacing[0]), float(first_ds.PixelSpacing[1])]

    slice_thickness = 1.0
    if hasattr(first_ds, 'SliceThickness'):
        slice_thickness = float(first_ds.SliceThickness)
    elif len(dicom_files) > 1:
        ds1, ds2 = dicom_files[0][1], dicom_files[1][1]
        if hasattr(ds1, 'SliceLocation') and hasattr(ds2, 'SliceLocation'):
            slice_thickness = abs(float(ds2.SliceLocation) - float(ds1.SliceLocation))

    spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])
    print(f"Voxel spacing: {spacing}")

    volume = np.zeros((num_slices, rows, cols), dtype=np.int16)

    for i, (filepath, ds) in enumerate(dicom_files):
        pixel_array = ds.pixel_array
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        pixel_array = pixel_array * slope + intercept
        volume[i, :, :] = pixel_array

        if (i + 1) % 100 == 0:
            print(f"Loaded {i + 1}/{num_slices} slices...")

    print(f"Volume loaded. Shape: {volume.shape}, Range: [{volume.min()}, {volume.max()}]")

    return volume, spacing

def load_volume(directory):
    """Auto-detect format and load volume."""
    img_type, count = detect_image_type(directory)

    if img_type == 'dicom':
        return load_dicom_series(directory)
    elif img_type in ('jpg', 'png'):
        return load_image_series(directory)
    else:
        raise ValueError(f"No supported image files found in {directory}")

def create_mesh(volume, spacing, preset_name='auto'):
    """Create 3D mesh from volume using preset settings."""
    print("Creating 3D mesh...")

    preset = SCAN_PRESETS.get(preset_name, SCAN_PRESETS['auto'])
    threshold = preset['threshold']
    keep_n_largest = preset['keep_n_largest']
    downsample_threshold = preset['downsample_threshold']
    use_morphological_closing = preset['use_morphological_closing']

    # Get data range
    vol_min, vol_max = volume.min(), volume.max()

    # Auto-detect threshold if not set or outside data range
    if threshold is None or threshold >= vol_max or threshold <= vol_min:
        positive_vals = volume[volume > 0]
        if len(positive_vals) > 0:
            threshold = np.percentile(positive_vals, 70)
        else:
            threshold = np.percentile(volume, 80)
        print(f"Auto-detected threshold: {threshold:.1f} (data range: [{vol_min}, {vol_max}])")
    else:
        print(f"Using threshold: {threshold} (preset: {preset_name}, data range: [{vol_min}, {vol_max}])")

    # Downsample if needed
    step = 1
    vol_size = volume.shape[0] * volume.shape[1] * volume.shape[2]
    if vol_size > downsample_threshold:
        step = 2
        volume = volume[::step, ::step, ::step]
        spacing = tuple(s * step for s in spacing)
        print(f"Downsampled to: {volume.shape}")

    # Create binary mask
    binary = volume > threshold

    # Connected component analysis
    print("  Finding connected components...")
    labeled, num_features = ndimage.label(binary)

    if num_features > 0:
        component_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        sorted_indices = np.argsort(component_sizes)[::-1]
        keep_labels = sorted_indices[:keep_n_largest] + 1
        binary = np.isin(labeled, keep_labels)
    else:
        # No components found - try a lower threshold
        print("  Warning: No voxels above threshold, trying lower threshold...")
        threshold = np.percentile(volume, 60)
        print(f"  Retrying with threshold: {threshold:.1f}")
        binary = volume > threshold
        labeled, num_features = ndimage.label(binary)
        if num_features > 0:
            component_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
            sorted_indices = np.argsort(component_sizes)[::-1]
            keep_labels = sorted_indices[:keep_n_largest] + 1
            binary = np.isin(labeled, keep_labels)

    # Check if we have any voxels to mesh
    if not binary.any():
        print("  Error: No voxels found for mesh generation")
        return None, None, None

    # Smooth and clean
    print("  Smoothing...")
    binary = morphology.remove_small_holes(binary, area_threshold=500)

    # Apply morphological closing for dental/bone scans (cleaner surface)
    if use_morphological_closing:
        print("  Applying morphological closing...")
        struct = morphology.ball(1)
        binary = ndimage.binary_closing(binary, structure=struct, iterations=1)

    smoothed = ndimage.gaussian_filter(binary.astype(np.float32), sigma=1.0)

    # Marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes(
            smoothed, level=0.5, spacing=spacing, step_size=1
        )
    except Exception as e:
        print(f"  Mesh generation failed: {e}")
        return None, None, None

    print(f"Mesh created: {len(verts)} vertices, {len(faces)} faces")

    # Center and scale
    center = verts.mean(axis=0)
    verts = verts - center
    max_extent = np.abs(verts).max()
    if max_extent > 0:
        verts = verts / max_extent * 100

    return verts, faces, normals

def get_slice_image(volume, axis, index, window_center=0, window_width=1000):
    """Extract 2D slice as base64 PNG."""
    if axis == 'axial':
        index = max(0, min(index, volume.shape[0] - 1))
        slice_data = volume[index, :, :]
    elif axis == 'sagittal':
        index = max(0, min(index, volume.shape[2] - 1))
        slice_data = volume[:, :, index]
    elif axis == 'coronal':
        index = max(0, min(index, volume.shape[1] - 1))
        slice_data = volume[:, index, :]
    else:
        return None

    min_val = window_center - window_width // 2
    max_val = window_center + window_width // 2
    slice_data = np.clip(slice_data, min_val, max_val)
    slice_data = ((slice_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    img = Image.fromarray(slice_data)
    max_dim = 400
    ratio = max_dim / max(img.size)
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

# ============== HTML Templates ==============

DIRECTORY_PICKER_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>CT Viewer - Select Directory</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            padding: 40px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            font-family: Arial, sans-serif;
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        h1 {
            color: #4fc3f7;
            margin-bottom: 30px;
        }
        .section {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 25px;
        }
        .section h2 {
            color: #4fc3f7;
            margin-top: 0;
            font-size: 18px;
            border-bottom: 1px solid rgba(79,195,247,0.3);
            padding-bottom: 10px;
        }
        .current-path {
            background: rgba(0,0,0,0.3);
            padding: 12px 15px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 13px;
            margin-bottom: 15px;
            word-break: break-all;
        }
        .folder-list {
            list-style: none;
            padding: 0;
            margin: 0;
            max-height: 400px;
            overflow-y: auto;
        }
        .folder-list li {
            padding: 10px 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
            margin-bottom: 6px;
            cursor: pointer;
            transition: background 0.2s;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .folder-list li:hover { background: rgba(79,195,247,0.2); }
        .folder-list li.has-images {
            background: rgba(76, 175, 80, 0.2);
            border-left: 3px solid #4caf50;
        }
        .folder-list li.has-images:hover { background: rgba(76, 175, 80, 0.35); }
        .folder-list li.parent-dir {
            background: rgba(255,193,7,0.15);
            font-weight: bold;
        }
        .folder-list li.parent-dir:hover { background: rgba(255,193,7,0.3); }
        .folder-name {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .folder-icon { font-size: 18px; }
        .image-count {
            background: #4caf50;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: bold;
        }
        button {
            padding: 12px 25px;
            background: #4fc3f7;
            color: #1a1a2e;
            border: none;
            border-radius: 6px;
            font-weight: bold;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover { background: #81d4fa; }
        button:disabled {
            background: #555;
            color: #888;
            cursor: not-allowed;
        }
        .open-btn {
            width: 100%;
            padding: 15px;
            font-size: 16px;
            margin-top: 15px;
        }
        .open-btn.ready {
            background: #4caf50;
        }
        .open-btn.ready:hover { background: #66bb6a; }
        .recent-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .recent-list li {
            padding: 10px 12px;
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 10px;
            transition: background 0.2s;
        }
        .recent-list li:hover { background: rgba(79,195,247,0.2); }
        .recent-path {
            flex: 1;
            cursor: pointer;
            font-family: monospace;
            font-size: 12px;
            word-break: break-all;
        }
        .open-recent-btn {
            padding: 6px 14px;
            background: #4caf50;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            cursor: pointer;
            white-space: nowrap;
        }
        .open-recent-btn:hover { background: #66bb6a; }
        .empty-message {
            color: rgba(255,255,255,0.5);
            font-style: italic;
        }
        .info-box {
            background: rgba(79,195,247,0.1);
            border-left: 3px solid #4fc3f7;
            padding: 15px;
            border-radius: 0 6px 6px 0;
            font-size: 13px;
            line-height: 1.6;
            margin-top: 15px;
        }
        .error { color: #ff6b6b; margin-top: 10px; }
        .status-text {
            text-align: center;
            padding: 10px;
            color: #aaa;
            font-size: 13px;
        }
        /* Loading Modal */
        .loading-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.85);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .loading-modal.show { display: flex; }
        .loading-content {
            background: #16213e;
            padding: 40px 60px;
            border-radius: 12px;
            text-align: center;
            border: 2px solid #4fc3f7;
        }
        .spinner {
            width: 60px;
            height: 60px;
            border: 4px solid rgba(79, 195, 247, 0.3);
            border-top-color: #4fc3f7;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .loading-text {
            color: #4fc3f7;
            font-size: 18px;
            font-weight: bold;
        }
        .loading-subtext {
            color: #aaa;
            font-size: 13px;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>CT Viewer</h1>

        {% if error %}
        <p class="error">{{ error }}</p>
        {% endif %}

        {% if recent %}
        <div class="section">
            <h2>Recent Folders</h2>
            <ul class="recent-list">
                {% for path in recent %}
                <li>
                    <span class="recent-path" onclick="navigateTo('{{ path }}')">{{ path }}</span>
                    <button class="open-recent-btn" onclick="openRecent('{{ path }}')">Open</button>
                </li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        <div class="section">
            <h2>Browse Folders</h2>
            <div class="current-path">{{ current_path }}</div>

            <ul class="folder-list">
                {% if parent_path %}
                <li class="parent-dir" onclick="navigateTo('{{ parent_path }}')">
                    <span class="folder-name"><span class="folder-icon">⬆</span> ..</span>
                </li>
                {% endif %}
                {% for folder in folders %}
                <li class="{{ 'has-images' if folder.has_dcm else '' }}" onclick="navigateTo('{{ folder.path }}')">
                    <span class="folder-name">
                        <span class="folder-icon">📁</span>
                        {{ folder.name }}
                    </span>
                    {% if folder.dcm_count > 0 %}
                    <span class="image-count">{{ folder.dcm_count }} images</span>
                    {% endif %}
                </li>
                {% endfor %}
            </ul>

            {% if dcm_count > 0 %}
            <p class="status-text">This folder contains {{ dcm_count }} image files</p>
            {% endif %}

            <form id="load-form" action="/load" method="post">
                <input type="hidden" name="path" id="load-path" value="{{ current_path }}">
                <button type="submit" class="open-btn {{ 'ready' if dcm_count > 0 else '' }}"
                        {{ '' if dcm_count > 0 else 'disabled' }}
                        onclick="showLoading(event)">
                    {% if dcm_count > 0 %}
                    Open This Folder ({{ dcm_count }} images)
                    {% else %}
                    Select a folder containing images
                    {% endif %}
                </button>
            </form>

            <div class="info-box">
                <b>Supported formats:</b><br>
                • DICOM files (.dcm) - Medical CT/MRI scans<br>
                • JPG sequences (.jpg, .jpeg) - Exported CT slices<br>
                • PNG sequences (.png) - Exported CT slices<br><br>
                <b>Tip:</b> Folders with images are highlighted in green.
            </div>
        </div>
    </div>

    <!-- Loading Modal -->
    <div id="loading-modal" class="loading-modal">
        <div class="loading-content">
            <div class="spinner"></div>
            <div class="loading-text">Loading CT Images...</div>
            <div class="loading-subtext">Building 3D model, please wait</div>
        </div>
    </div>

    <script>
        function navigateTo(path) {
            window.location.href = '/browse?path=' + encodeURIComponent(path);
        }

        function openRecent(path) {
            showLoadingModal();
            document.getElementById('load-path').value = path;
            document.getElementById('load-form').submit();
        }

        function showLoading(event) {
            showLoadingModal();
        }

        function showLoadingModal() {
            document.getElementById('loading-modal').classList.add('show');
        }
    </script>

    <!-- Medical Disclaimer -->
    <div style="margin-top: 40px; padding: 20px; background: rgba(255,100,100,0.1); border: 1px solid rgba(255,100,100,0.3); border-radius: 8px; max-width: 900px; margin-left: auto; margin-right: auto;">
        <h3 style="color: #ff6b6b; margin: 0 0 10px 0; font-size: 14px;">Medical Disclaimer</h3>
        <p style="font-size: 12px; color: rgba(255,255,255,0.7); margin: 0; line-height: 1.6;">
            <strong>THIS SOFTWARE IS NOT A MEDICAL DEVICE.</strong> It is provided for educational and informational purposes only.
            This software has not been reviewed or approved by the FDA or any regulatory body. It is NOT intended for clinical diagnosis,
            treatment planning, or medical decision-making. The 3D renderings may contain artifacts or inaccuracies.
            Always consult qualified healthcare professionals for interpretation of medical images.
            <br><br>
            THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND. By using this software, you acknowledge these limitations.
        </p>
    </div>
</body>
</html>
'''

VIEWER_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>CT Viewer - {{ directory_name }}</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            overflow: hidden;
            background: #1a1a2e;
            font-family: Arial, sans-serif;
            color: white;
        }
        #container { display: flex; height: 100vh; }
        #viewer3d { flex: 1; position: relative; }
        #sidebar {
            width: 500px;
            background: #16213e;
            padding: 15px;
            overflow-y: auto;
            border-left: 2px solid #4fc3f7;
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.85);
            padding: 15px;
            border-radius: 8px;
            z-index: 100;
            max-width: 300px;
        }
        #info h2 { margin-top: 0; color: #4fc3f7; font-size: 16px; }
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 24px;
            z-index: 200;
        }
        .back-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 100;
            padding: 8px 15px;
            background: rgba(79,195,247,0.2);
            border: 1px solid #4fc3f7;
            color: #4fc3f7;
            border-radius: 6px;
            cursor: pointer;
            text-decoration: none;
            font-size: 13px;
        }
        .back-btn:hover { background: rgba(79,195,247,0.4); }
        .slice-panel {
            margin-bottom: 15px;
            background: #0f3460;
            padding: 12px;
            border-radius: 8px;
        }
        .slice-panel h3 {
            margin: 0 0 8px 0;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .plane-color {
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }
        .slice-panel img {
            width: 100%;
            border-radius: 4px;
            background: black;
        }
        .slider-container { margin-top: 10px; }
        .slider-container input { width: 100%; }
        .slider-label {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #aaa;
        }
        h2.sidebar-title {
            color: #4fc3f7;
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 18px;
        }
        .controls-section {
            background: #0f3460;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .controls-section h3 {
            margin: 0 0 10px 0;
            color: #4fc3f7;
            font-size: 13px;
        }
        .controls-section label {
            display: block;
            margin-bottom: 3px;
            font-size: 11px;
        }
        .controls-section input[type="range"] {
            width: 100%;
            margin-bottom: 8px;
        }
        .current-slice {
            background: #4fc3f7;
            color: #1a1a2e;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 12px;
        }
        .dir-info {
            font-size: 11px;
            color: #aaa;
            margin-bottom: 15px;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 6px;
        }
        .scan-type-badge {
            display: inline-block;
            background: #4fc3f7;
            color: #1a1a2e;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 10px;
        }
        .view-buttons {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }
        .view-btn {
            padding: 6px 12px;
            background: rgba(79,195,247,0.2);
            border: 1px solid #4fc3f7;
            color: #4fc3f7;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.2s;
        }
        .view-btn:hover {
            background: rgba(79,195,247,0.4);
        }
        .view-btn.primary {
            background: #4fc3f7;
            color: #1a1a2e;
            font-weight: bold;
        }
        .view-btn.primary:hover {
            background: #81d4fa;
        }
        .preset-select {
            width: 100%;
            padding: 8px;
            background: rgba(255,255,255,0.1);
            border: 1px solid #4fc3f7;
            color: white;
            border-radius: 4px;
            font-size: 12px;
            margin-bottom: 10px;
        }
        .preset-select option {
            background: #16213e;
            color: white;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="viewer3d">
            <div id="loading">Loading 3D model...</div>
            <a href="/" class="back-btn">← Change Directory</a>
            <div id="info">
                <h2>CT Viewer</h2>
                <p style="font-size: 12px; margin: 8px 0;">
                    <b>3D Controls:</b><br>
                    • Left-click + drag: Rotate<br>
                    • Right-click + drag: Pan<br>
                    • Scroll: Zoom<br>
                    • Right-click model: Set planes
                </p>
            </div>
        </div>
        <div id="sidebar">
            <h2 class="sidebar-title">Cross-Sectional Views</h2>

            <div class="dir-info">
                <b>Directory:</b> {{ directory_name }}<br>
                <b>Format:</b> {{ image_type | upper }}<br>
                <b>Slices:</b> {{ slice_count }}<br>
                <b>Detected Type:</b> <span class="scan-type-badge">{{ scan_type }}</span>
            </div>

            <div class="controls-section">
                <h3>3D View Controls</h3>
                <div class="view-buttons">
                    <button class="view-btn primary" onclick="resetView()">Reset View</button>
                    <button class="view-btn" onclick="setView('front')">Front</button>
                    <button class="view-btn" onclick="setView('back')">Back</button>
                    <button class="view-btn" onclick="setView('left')">Left</button>
                    <button class="view-btn" onclick="setView('right')">Right</button>
                    <button class="view-btn" onclick="setView('top')">Top</button>
                </div>
                <p style="font-size: 11px; color: #aaa; margin: 5px 0 0 0;">Click a view or drag to rotate freely</p>
            </div>

            <div class="controls-section">
                <h3>Image Contrast</h3>
                <label>Preset:</label>
                <select class="preset-select" id="preset-select" onchange="applyPreset(this.value)">
                    <option value="custom" {% if scan_type == 'auto' %}selected{% endif %}>Custom</option>
                    <option value="dental" {% if scan_type == 'dental' %}selected{% endif %}>Dental / Bone (HU 400/2000)</option>
                    <option value="bone" {% if scan_type == 'bone' %}selected{% endif %}>Bone (HU 400/2000)</option>
                    <option value="soft_tissue">Soft Tissue (HU 40/400)</option>
                    <option value="lung" {% if scan_type == 'lung' %}selected{% endif %}>Lung (HU -600/1500)</option>
                </select>
                <label>Brightness: <span id="wc-val">{{ window_center }}</span></label>
                <input type="range" id="window-center" min="-1000" max="2000" value="{{ window_center }}">
                <label>Contrast: <span id="ww-val">{{ window_width }}</span></label>
                <input type="range" id="window-width" min="100" max="4000" value="{{ window_width }}">
            </div>

            <div class="slice-panel">
                <h3>
                    <span class="plane-color" style="background: #4fc3f7;"></span>
                    Axial View
                    <span class="current-slice" id="axial-pos">0</span>
                </h3>
                <img id="axial-img" src="" alt="Axial">
                <div class="slider-container">
                    <input type="range" id="axial-slider" min="0" max="100" value="50">
                    <div class="slider-label"><span>Bottom</span><span>Top</span></div>
                </div>
            </div>

            <div class="slice-panel">
                <h3>
                    <span class="plane-color" style="background: #ff6b6b;"></span>
                    Sagittal View
                    <span class="current-slice" id="sagittal-pos">0</span>
                </h3>
                <img id="sagittal-img" src="" alt="Sagittal">
                <div class="slider-container">
                    <input type="range" id="sagittal-slider" min="0" max="100" value="50">
                    <div class="slider-label"><span>Left</span><span>Right</span></div>
                </div>
            </div>

            <div class="slice-panel">
                <h3>
                    <span class="plane-color" style="background: #6bff6b;"></span>
                    Coronal View
                    <span class="current-slice" id="coronal-pos">0</span>
                </h3>
                <img id="coronal-img" src="" alt="Coronal">
                <div class="slider-container">
                    <input type="range" id="coronal-slider" min="0" max="100" value="50">
                    <div class="slider-label"><span>Back</span><span>Front</span></div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

    <script>
        let scene, camera, renderer, controls, mesh;
        let volumeShape = {{ shape | tojson }};
        let windowCenter = {{ window_center }};
        let windowWidth = {{ window_width }};
        let meshColor = {{ mesh_color }};
        let raycaster, mouse;
        let axialPlane, sagittalPlane, coronalPlane;

        // Preset definitions for window/level
        const presets = {
            dental: { wc: 400, ww: 2000 },
            bone: { wc: 400, ww: 2000 },
            soft_tissue: { wc: 40, ww: 400 },
            lung: { wc: -600, ww: 1500 },
            custom: null
        };

        function init() {
            const container = document.getElementById('viewer3d');
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);

            camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(0, 0, 200);

            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.target.set(0, 0, 0);  // Ensure rotation around center

            scene.add(new THREE.AmbientLight(0x404040, 0.5));
            const light1 = new THREE.DirectionalLight(0xffffff, 0.8);
            light1.position.set(1, 1, 1);
            scene.add(light1);
            const light2 = new THREE.DirectionalLight(0xffffff, 0.5);
            light2.position.set(-1, -1, -1);
            scene.add(light2);

            raycaster = new THREE.Raycaster();
            mouse = new THREE.Vector2();
            renderer.domElement.addEventListener('contextmenu', onRightClick);

            loadMesh();
            setupSliders();

            window.addEventListener('resize', onWindowResize);
            animate();
        }

        // View control functions
        function resetView() {
            // Reset camera to default front-facing position
            camera.position.set(0, 0, 200);
            camera.up.set(0, 1, 0);
            controls.target.set(0, 0, 0);
            controls.update();
        }

        function setView(view) {
            const distance = 200;
            controls.target.set(0, 0, 0);

            switch(view) {
                case 'front':
                    camera.position.set(0, 0, distance);
                    camera.up.set(0, 1, 0);
                    break;
                case 'back':
                    camera.position.set(0, 0, -distance);
                    camera.up.set(0, 1, 0);
                    break;
                case 'left':
                    camera.position.set(-distance, 0, 0);
                    camera.up.set(0, 1, 0);
                    break;
                case 'right':
                case 'side':  // Keep 'side' for backward compatibility
                    camera.position.set(distance, 0, 0);
                    camera.up.set(0, 1, 0);
                    break;
                case 'top':
                    camera.position.set(0, distance, 0);
                    camera.up.set(0, 0, -1);
                    break;
                case 'bottom':
                    camera.position.set(0, -distance, 0);
                    camera.up.set(0, 0, 1);
                    break;
            }
            controls.update();
        }

        function applyPreset(presetName) {
            if (presetName === 'custom') return;

            const preset = presets[presetName];
            if (preset) {
                windowCenter = preset.wc;
                windowWidth = preset.ww;
                document.getElementById('window-center').value = windowCenter;
                document.getElementById('window-width').value = windowWidth;
                document.getElementById('wc-val').textContent = windowCenter;
                document.getElementById('ww-val').textContent = windowWidth;
                updateAllSlices();
            }
        }

        function onRightClick(event) {
            event.preventDefault();
            if (!mesh || !volumeShape) return;

            const container = document.getElementById('viewer3d');
            const rect = container.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / container.clientWidth) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / container.clientHeight) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(mesh);

            if (intersects.length > 0) {
                const point = intersects[0].point;
                const rotatedPoint = point.clone();
                rotatedPoint.applyAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 6);

                const axialIdx = Math.max(0, Math.min(volumeShape[0] - 1,
                    Math.round((rotatedPoint.z / 100 + 0.5) * volumeShape[0])));
                const coronalIdx = Math.max(0, Math.min(volumeShape[1] - 1,
                    Math.round((rotatedPoint.y / 100 + 0.5) * volumeShape[1])));
                const sagittalIdx = Math.max(0, Math.min(volumeShape[2] - 1,
                    Math.round((rotatedPoint.x / 100 + 0.5) * volumeShape[2])));

                document.getElementById('axial-slider').value = axialIdx;
                document.getElementById('coronal-slider').value = coronalIdx;
                document.getElementById('sagittal-slider').value = sagittalIdx;

                updateSlice('axial', axialIdx);
                updateSlice('coronal', coronalIdx);
                updateSlice('sagittal', sagittalIdx);
                updatePlaneHelpers(axialIdx, coronalIdx, sagittalIdx);
            }
        }

        function updatePlaneHelpers(axial, coronal, sagittal) {
            if (axialPlane) scene.remove(axialPlane);
            if (sagittalPlane) scene.remove(sagittalPlane);
            if (coronalPlane) scene.remove(coronalPlane);

            const size = 120;

            const axialMat = new THREE.MeshBasicMaterial({color: 0x4fc3f7, transparent: true, opacity: 0.15, side: THREE.DoubleSide});
            axialPlane = new THREE.Mesh(new THREE.PlaneGeometry(size, size), axialMat);
            axialPlane.rotation.x = Math.PI / 2 - Math.PI / 6;
            axialPlane.position.z = (axial / volumeShape[0] - 0.5) * 100 * Math.cos(Math.PI/6);
            axialPlane.position.y = -(axial / volumeShape[0] - 0.5) * 100 * Math.sin(Math.PI/6);
            scene.add(axialPlane);

            const sagMat = new THREE.MeshBasicMaterial({color: 0xff6b6b, transparent: true, opacity: 0.15, side: THREE.DoubleSide});
            sagittalPlane = new THREE.Mesh(new THREE.PlaneGeometry(size, size), sagMat);
            sagittalPlane.rotation.y = Math.PI / 2;
            sagittalPlane.rotation.x = -Math.PI / 6;
            sagittalPlane.position.x = (sagittal / volumeShape[2] - 0.5) * 100;
            scene.add(sagittalPlane);

            const corMat = new THREE.MeshBasicMaterial({color: 0x6bff6b, transparent: true, opacity: 0.15, side: THREE.DoubleSide});
            coronalPlane = new THREE.Mesh(new THREE.PlaneGeometry(size, size), corMat);
            coronalPlane.rotation.x = -Math.PI / 6;
            coronalPlane.position.y = (coronal / volumeShape[1] - 0.5) * 100 * Math.cos(Math.PI/6);
            coronalPlane.position.z = (coronal / volumeShape[1] - 0.5) * 100 * Math.sin(Math.PI/6);
            scene.add(coronalPlane);
        }

        function loadMesh() {
            fetch('/mesh_data')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('loading').style.display = 'none';
                    if (data.error) return;

                    const geometry = new THREE.BufferGeometry();
                    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(data.vertices), 3));
                    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(data.faces), 1));
                    geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(data.normals), 3));

                    // Use mesh color from preset (passed from server, or from data)
                    const color = data.color || meshColor || 0xcccccc;

                    const material = new THREE.MeshPhongMaterial({
                        color: color,
                        specular: 0x222222,
                        shininess: 25,
                        side: THREE.DoubleSide
                    });

                    mesh = new THREE.Mesh(geometry, material);
                    // Flip mesh so head/top is at top (rotate 180 degrees around X)
                    // Then tilt slightly for better 3D viewing
                    mesh.rotation.x = Math.PI - Math.PI / 6;

                    // Center the mesh properly
                    geometry.computeBoundingBox();
                    const box = geometry.boundingBox;
                    const center = new THREE.Vector3();
                    box.getCenter(center);
                    geometry.translate(-center.x, -center.y, -center.z);

                    scene.add(mesh);

                    // Set camera to look at center
                    controls.target.set(0, 0, 0);
                    controls.update();

                    // Initialize planes and slices
                    const mid = [Math.floor(volumeShape[0]/2), Math.floor(volumeShape[1]/2), Math.floor(volumeShape[2]/2)];
                    document.getElementById('axial-slider').max = volumeShape[0] - 1;
                    document.getElementById('sagittal-slider').max = volumeShape[2] - 1;
                    document.getElementById('coronal-slider').max = volumeShape[1] - 1;
                    document.getElementById('axial-slider').value = mid[0];
                    document.getElementById('sagittal-slider').value = mid[2];
                    document.getElementById('coronal-slider').value = mid[1];
                    updateAllSlices();
                    updatePlaneHelpers(mid[0], mid[1], mid[2]);
                });
        }

        function setupSliders() {
            ['axial', 'sagittal', 'coronal'].forEach(axis => {
                document.getElementById(axis + '-slider').addEventListener('input', function() {
                    updateSlice(axis, this.value);
                    updatePlaneHelpers(
                        parseInt(document.getElementById('axial-slider').value),
                        parseInt(document.getElementById('coronal-slider').value),
                        parseInt(document.getElementById('sagittal-slider').value)
                    );
                });
            });

            document.getElementById('window-center').addEventListener('input', function() {
                windowCenter = parseInt(this.value);
                document.getElementById('wc-val').textContent = windowCenter;
                document.getElementById('preset-select').value = 'custom';  // Switch to custom when manually adjusted
                updateAllSlices();
            });
            document.getElementById('window-width').addEventListener('input', function() {
                windowWidth = parseInt(this.value);
                document.getElementById('ww-val').textContent = windowWidth;
                document.getElementById('preset-select').value = 'custom';  // Switch to custom when manually adjusted
                updateAllSlices();
            });
        }

        function updateSlice(axis, index) {
            document.getElementById(axis + '-pos').textContent = index;
            fetch(`/slice/${axis}/${index}?wc=${windowCenter}&ww=${windowWidth}`)
                .then(r => r.json())
                .then(data => {
                    document.getElementById(axis + '-img').src = 'data:image/png;base64,' + data.image;
                });
        }

        function updateAllSlices() {
            updateSlice('axial', document.getElementById('axial-slider').value);
            updateSlice('sagittal', document.getElementById('sagittal-slider').value);
            updateSlice('coronal', document.getElementById('coronal-slider').value);
        }

        function onWindowResize() {
            const container = document.getElementById('viewer3d');
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }

        init();
    </script>
</body>
</html>
'''

# ============== Routes ==============

@app.route('/')
def index():
    """Redirect to folder browser."""
    return redirect(url_for('browse'))

@app.route('/browse')
def browse():
    """Show folder browser."""
    path = request.args.get('path', '')
    if not path:
        # Default to parent of script directory
        path = str(Path(__file__).parent.parent)
    if not os.path.isdir(path):
        path = str(Path.home())

    parent_path = str(Path(path).parent) if path != '/' else None
    folders, dcm_count = get_folder_info(path)
    recent = load_recent_directories()

    return render_template_string(
        DIRECTORY_PICKER_HTML,
        current_path=path,
        parent_path=parent_path,
        folders=folders,
        dcm_count=dcm_count,
        recent=recent,
        error=None
    )

@app.route('/load', methods=['POST'])
def load_directory():
    global current_volume, current_spacing, current_shape, current_mesh, current_directory, current_scan_type

    path = request.form.get('path', '').strip()

    def error_response(error_msg, browse_path=None):
        if not browse_path:
            browse_path = str(Path(__file__).parent.parent)
        parent_path = str(Path(browse_path).parent) if browse_path != '/' else None
        folders, dcm_count = get_folder_info(browse_path)
        return render_template_string(
            DIRECTORY_PICKER_HTML,
            current_path=browse_path,
            parent_path=parent_path,
            folders=folders,
            dcm_count=dcm_count,
            recent=load_recent_directories(),
            error=error_msg
        )

    if not path:
        return error_response("Please enter a path")

    if not os.path.isdir(path):
        return error_response(f"Directory not found: {path}")

    img_type, count = detect_image_type(path)
    if img_type is None:
        return error_response("No supported image files found (DICOM, JPG, or PNG)", path)

    # Load the volume
    try:
        current_volume, current_spacing = load_volume(path)
        current_shape = list(current_volume.shape)
        current_directory = path

        # Detect scan type for optimal settings
        dicom_meta = current_dicom_metadata if img_type == 'dicom' else None
        current_scan_type = detect_scan_type(path, current_shape, dicom_meta)

        # Create mesh with detected preset
        verts, faces, normals = create_mesh(current_volume, current_spacing, preset_name=current_scan_type)
        if verts is not None:
            preset = SCAN_PRESETS.get(current_scan_type, SCAN_PRESETS['auto'])
            current_mesh = {
                'vertices': verts.flatten().tolist(),
                'faces': faces.flatten().tolist(),
                'normals': normals.flatten().tolist(),
                'color': preset['mesh_color']
            }
        else:
            current_mesh = None

        # Save to recent
        add_recent_directory(path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return error_response(f"Error loading: {str(e)}", path)

    return redirect(url_for('viewer'))

@app.route('/viewer')
def viewer():
    if current_volume is None:
        return redirect(url_for('index'))

    img_type, count = detect_image_type(current_directory)
    dir_name = os.path.basename(current_directory)

    # Get preset settings for detected scan type
    preset = SCAN_PRESETS.get(current_scan_type, SCAN_PRESETS['auto'])

    # Determine window settings - use preset if available, otherwise auto-calculate
    if preset['window_center'] is not None:
        window_center = preset['window_center']
        window_width = preset['window_width']
    else:
        vol_min, vol_max = int(current_volume.min()), int(current_volume.max())
        window_center = (vol_min + vol_max) // 2
        window_width = vol_max - vol_min

    mesh_color = preset['mesh_color']

    return render_template_string(VIEWER_HTML,
                                  directory_name=dir_name,
                                  image_type=img_type,
                                  slice_count=count,
                                  shape=current_shape,
                                  window_center=window_center,
                                  window_width=window_width,
                                  scan_type=current_scan_type,
                                  mesh_color=mesh_color)

@app.route('/mesh_data')
def mesh_data():
    if current_mesh is None:
        return jsonify({'error': 'No mesh available'})
    return jsonify(current_mesh)

@app.route('/slice/<axis>/<int:index>')
def get_slice(axis, index):
    if current_volume is None:
        return jsonify({'error': 'No volume loaded'})

    wc = request.args.get('wc', 0, type=int)
    ww = request.args.get('ww', 1000, type=int)

    img = get_slice_image(current_volume, axis, index, wc, ww)
    return jsonify({'image': img})

def main():
    print("=" * 60)
    print("CT Viewer - Universal CT/MRI Image Viewer")
    print("=" * 60)
    print(f"Open http://localhost:{PORT} in your browser")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)

if __name__ == '__main__':
    main()
