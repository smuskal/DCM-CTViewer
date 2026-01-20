#!/usr/bin/env python3
"""
3D CT/MRI Viewer for Dental and Medical Scans
Web-based viewer using Flask and Three.js
Supports DICOM files and JPG/PNG image sequences
Features:
- Directory picker with recent history
- 3D reconstruction with noise reduction
- Cross-sectional MPR views with interactive plane selection
Run this script and open http://localhost:7001 in your browser
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

# Global variables
mesh_data = None
volume_data = None
volume_spacing = None
volume_shape = None
current_directory = None


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

def natural_sort_key(s):
    """Sort strings with numbers naturally (1, 2, 10 instead of 1, 10, 2)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]

def detect_image_type(directory):
    """Detect whether directory contains DICOM, JPG, or PNG files."""
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

    # Assume isotropic spacing for JPG/PNG (can be adjusted)
    spacing = (1.0, 1.0, 1.0)

    print(f"Volume loaded. Shape: {volume.shape}, Range: [{volume.min()}, {volume.max()}]")

    # No DICOM orientation for image series
    return volume, spacing, None

def extract_dicom_orientation(ds):
    """
    Extract orientation information from DICOM metadata.
    Returns dict with orientation info or None if not available.

    Key DICOM tags:
    - ImageOrientationPatient (0020,0037): Direction cosines of first row and column
    - PatientPosition (0018,5100): Patient position (HFS, HFP, FFS, FFP, etc.)
    - BodyPartExamined (0018,0015): Body part (HEAD, CHEST, ABDOMEN, etc.)
    """
    orientation = {}

    try:
        # ImageOrientationPatient: [row_x, row_y, row_z, col_x, col_y, col_z]
        if hasattr(ds, 'ImageOrientationPatient'):
            iop = [float(x) for x in ds.ImageOrientationPatient]
            orientation['imageOrientation'] = iop
            # Row direction = [iop[0], iop[1], iop[2]]
            # Column direction = [iop[3], iop[4], iop[5]]
            # Slice direction = cross product
            row_dir = np.array(iop[0:3])
            col_dir = np.array(iop[3:6])
            slice_dir = np.cross(row_dir, col_dir)
            orientation['rowDirection'] = row_dir.tolist()
            orientation['colDirection'] = col_dir.tolist()
            orientation['sliceDirection'] = slice_dir.tolist()
            print(f"  DICOM orientation - Row: {row_dir}, Col: {col_dir}, Slice: {slice_dir}")

        # PatientPosition: HFS (Head First Supine), HFP (Head First Prone),
        # FFS (Feet First Supine), FFP (Feet First Prone), etc.
        if hasattr(ds, 'PatientPosition'):
            orientation['patientPosition'] = str(ds.PatientPosition)
            print(f"  Patient position: {orientation['patientPosition']}")

        # BodyPartExamined
        if hasattr(ds, 'BodyPartExamined'):
            orientation['bodyPart'] = str(ds.BodyPartExamined)
            print(f"  Body part: {orientation['bodyPart']}")

        # Modality (CT, MR, etc.)
        if hasattr(ds, 'Modality'):
            orientation['modality'] = str(ds.Modality)

    except Exception as e:
        print(f"  Warning: Could not extract DICOM orientation: {e}")
        return None

    return orientation if orientation else None


def load_dicom_series(directory):
    """Load all DICOM files from a directory and return a 3D numpy array."""
    if not DICOM_AVAILABLE:
        raise ImportError("pydicom not available - install with: pip install pydicom")

    print(f"Loading DICOM files from: {directory}")

    dicom_files = []
    for f in os.listdir(directory):
        if f.endswith('.dcm'):
            filepath = os.path.join(directory, f)
            try:
                ds = pydicom.dcmread(filepath)
                if hasattr(ds, 'pixel_array'):
                    dicom_files.append((filepath, ds))
            except Exception as e:
                pass

    if not dicom_files:
        raise ValueError("No valid DICOM files found")

    print(f"Found {len(dicom_files)} DICOM files with pixel data")

    def get_sort_key(item):
        ds = item[1]
        if hasattr(ds, 'SliceLocation'):
            return float(ds.SliceLocation)
        elif hasattr(ds, 'InstanceNumber'):
            return int(ds.InstanceNumber)
        else:
            import re
            match = re.search(r'(\d+)', os.path.basename(item[0]))
            return int(match.group(1)) if match else 0

    dicom_files.sort(key=get_sort_key)

    first_ds = dicom_files[0][1]
    rows = first_ds.Rows
    cols = first_ds.Columns
    num_slices = len(dicom_files)

    print(f"Volume dimensions: {cols} x {rows} x {num_slices}")

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

    # Extract DICOM orientation metadata from first slice
    print("  Checking DICOM orientation metadata...")
    dicom_orientation = extract_dicom_orientation(first_ds)

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

    return volume, spacing, dicom_orientation

def load_volume(directory):
    """Auto-detect format and load volume."""
    img_type, count = detect_image_type(directory)

    if img_type == 'dicom':
        return load_dicom_series(directory)
    elif img_type in ('jpg', 'png'):
        return load_image_series(directory)
    else:
        raise ValueError(f"No supported image files found in {directory}")

def clean_volume_aggressive(volume, threshold, keep_n_largest=3):
    """Apply aggressive noise reduction - keep only largest connected components."""
    print("Applying noise reduction...")

    binary = volume > threshold

    print("  Labeling connected components...")
    labeled, num_features = ndimage.label(binary)
    print(f"  Found {num_features} connected components")

    if num_features == 0:
        return binary

    component_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    sorted_indices = np.argsort(component_sizes)[::-1]
    keep_labels = sorted_indices[:keep_n_largest] + 1

    print(f"  Keeping {keep_n_largest} largest components")
    clean_binary = np.isin(labeled, keep_labels)

    print("  Filling holes...")
    clean_binary = morphology.remove_small_holes(clean_binary, max_size=500)

    print("  Smoothing...")
    struct = morphology.ball(1)
    clean_binary = ndimage.binary_closing(clean_binary, structure=struct, iterations=1)

    return clean_binary

def create_mesh(volume, spacing, threshold=None, keep_n_largest=3):
    """Create a 3D mesh from the volume using marching cubes."""
    print("Creating 3D mesh...")

    # Auto-detect threshold if not provided or if it's outside the data range
    vol_min, vol_max = volume.min(), volume.max()
    if threshold is None or threshold >= vol_max or threshold <= vol_min:
        # Use percentile-based threshold for auto-detection
        positive_vals = volume[volume > 0]
        if len(positive_vals) > 0:
            threshold = np.percentile(positive_vals, 70)
        else:
            threshold = np.percentile(volume, 80)
        print(f"Auto-detected threshold: {threshold:.1f} (data range: [{vol_min}, {vol_max}])")
    else:
        print(f"Using threshold: {threshold} HU (data range: [{vol_min}, {vol_max}])")

    step = 1
    vol_size = volume.shape[0] * volume.shape[1] * volume.shape[2]
    if vol_size > 400 * 400 * 400:
        step = 2
        volume = volume[::step, ::step, ::step]
        spacing = tuple(s * step for s in spacing)
        print(f"Downsampled to: {volume.shape}")

    binary = clean_volume_aggressive(volume, threshold, keep_n_largest)

    # Check if we have any voxels to mesh
    if not binary.any():
        print("  Warning: No voxels above threshold, trying lower threshold...")
        # Try a lower percentile
        threshold = np.percentile(volume, 60)
        print(f"  Retrying with threshold: {threshold:.1f}")
        binary = clean_volume_aggressive(volume, threshold, keep_n_largest)
        if not binary.any():
            raise ValueError("Could not find suitable threshold for mesh generation")

    print("  Gaussian smoothing...")
    smoothed = ndimage.gaussian_filter(binary.astype(np.float32), sigma=1.0)

    try:
        verts, faces, normals, values = measure.marching_cubes(
            smoothed,
            level=0.5,
            spacing=spacing,
            step_size=1
        )
    except ValueError as e:
        raise ValueError(f"Mesh generation failed: {e}. Try adjusting the threshold.")

    print(f"Mesh created: {len(verts)} vertices, {len(faces)} faces")

    # Center based on VOLUME dimensions, not mesh vertex mean
    # This ensures cross-section planes (positioned by volume indices) align with mesh
    vol_extent = np.array([
        volume.shape[0] * spacing[0],
        volume.shape[1] * spacing[1],
        volume.shape[2] * spacing[2]
    ])
    vol_center = vol_extent / 2
    verts_centered = verts - vol_center

    # Scale uniformly based on largest volume dimension
    # This ensures planes at volume edges reach mesh edges
    max_vol_dim = vol_extent.max()
    scale_factor = 100 / max_vol_dim if max_vol_dim > 0 else 1
    verts_scaled = verts_centered * scale_factor

    return verts_scaled, faces, normals


def analyze_mesh_orientation(vertices, dicom_orientation=None):
    """
    Analyze mesh geometry to determine optimal viewing orientation.
    Returns principal axes and camera view positions for anatomical views.

    Standard DICOM/medical imaging convention (used regardless of bounding box):
    - Volume axis 0 (mesh X) = axial/superior-inferior = head-to-toe
    - Volume axis 1 (mesh Y) = coronal/anterior-posterior = front-to-back
    - Volume axis 2 (mesh Z) = sagittal/left-right = left-to-right

    This function determines front/back facing using DICOM metadata or asymmetry.
    """
    if len(vertices) == 0:
        return get_default_orientation()

    # Reshape vertices if flattened
    if vertices.ndim == 1:
        vertices = vertices.reshape(-1, 3)

    print("  Analyzing mesh orientation...")

    # Calculate bounding box for diagnostics
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    extents = max_coords - min_coords
    center = (min_coords + max_coords) / 2

    print(f"  Bounding box extents: X={extents[0]:.1f}, Y={extents[1]:.1f}, Z={extents[2]:.1f}")

    # Report aspect ratios for debugging
    aspect_xy = extents[0] / max(extents[1], 0.001)
    aspect_xz = extents[0] / max(extents[2], 0.001)
    print(f"  Aspect ratios: X/Y={aspect_xy:.2f}, X/Z={aspect_xz:.2f}")

    # Use DICOM orientation if available
    dicom_guided = False
    patient_facing = 'front'  # default: supine (face up)
    head_first = True  # default: head first

    if dicom_orientation:
        print("  Using DICOM orientation metadata...")
        dicom_guided = True

        # Check patient position
        if 'patientPosition' in dicom_orientation:
            pos = dicom_orientation['patientPosition'].upper()
            # HFS = Head First Supine, HFP = Head First Prone
            # FFS = Feet First Supine, FFP = Feet First Prone
            if pos.startswith('HF'):
                head_first = True
                print(f"    Patient position: Head First ({pos})")
            elif pos.startswith('FF'):
                head_first = False
                print(f"    Patient position: Feet First ({pos})")

            if 'P' in pos:
                patient_facing = 'back'
                print("    Patient is prone (face down)")
            elif 'S' in pos:
                patient_facing = 'front'
                print("    Patient is supine (face up)")

    # Detect front/back using asymmetry along Y axis (axis 1 = front-to-back)
    # The front of a body (face, chest) typically has more complex geometry
    axis_mid_y = center[1]
    front_half = vertices[vertices[:, 1] < axis_mid_y]
    back_half = vertices[vertices[:, 1] >= axis_mid_y]

    front_count = len(front_half)
    back_count = len(back_half)
    front_bias = front_count / max(front_count + back_count, 1)

    print(f"  Front/back distribution: {front_bias:.2%} on -Y side (axis 1)")

    # Determine if we need to flip the front direction
    # Use DICOM metadata if available, otherwise use geometric heuristic
    if dicom_guided:
        facing_away = (patient_facing == 'back')
        if facing_away:
            print("  DICOM indicates prone position (facing away)")
    else:
        # Geometric heuristic: if more vertices on +Y side, front is toward -Y
        # This is a weak heuristic and may not always be correct
        facing_away = front_bias < 0.45
        if facing_away:
            print("  Model may be facing away (geometric heuristic)")

    # Calculate the mesh rotation that was applied (Math.PI - Math.PI/6)
    # This rotation is around the X axis (head-to-toe axis)
    mesh_rotation_x = np.pi - np.pi / 6
    cos_r = np.cos(mesh_rotation_x)
    sin_r = np.sin(mesh_rotation_x)

    # Build rotation matrix for mesh rotation around X
    rot_matrix = np.array([
        [1, 0, 0],
        [0, cos_r, -sin_r],
        [0, sin_r, cos_r]
    ])

    # Define anatomical directions in mesh LOCAL coordinates
    # These produce the correct camera positions when transformed through rot_matrix:
    # - Front camera at (0, 173, -100)
    # - Back camera at (0, -173, 100)
    # - Left camera at (0, 100, 173)
    # - Right camera at (0, -100, -173)
    # - Top camera at (200, 0, 0)
    head_dir_local = np.array([1.0, 0.0, 0.0])   # +X = toward head
    front_dir_local = np.array([0.0, 1.0, 0.0])  # +Y = toward front (face)
    right_dir_local = np.array([0.0, 0.0, -1.0]) # -Z = toward right (DICOM convention: +Z is patient left)

    # Flip head direction if feet first
    if not head_first:
        head_dir_local = -head_dir_local
        print("  Flipped head direction for feet-first scan")

    # Apply mesh rotation to get WORLD coordinates
    head_dir_world = rot_matrix @ head_dir_local
    front_dir_world = rot_matrix @ front_dir_local
    right_dir_world = rot_matrix @ right_dir_local

    # If facing away (prone), flip the front direction
    if facing_away:
        front_dir_world = -front_dir_world
        print("  Flipped front direction for prone/facing-away")

    # Normalize directions
    head_dir_world = head_dir_world / np.linalg.norm(head_dir_world)
    front_dir_world = front_dir_world / np.linalg.norm(front_dir_world)
    right_dir_world = right_dir_world / np.linalg.norm(right_dir_world)

    print(f"  Head direction (world): [{head_dir_world[0]:.3f}, {head_dir_world[1]:.3f}, {head_dir_world[2]:.3f}]")
    print(f"  Front direction (world): [{front_dir_world[0]:.3f}, {front_dir_world[1]:.3f}, {front_dir_world[2]:.3f}]")
    print(f"  Right direction (world): [{right_dir_world[0]:.3f}, {right_dir_world[1]:.3f}, {right_dir_world[2]:.3f}]")

    # Calculate camera positions for each view (opposite to viewing direction)
    d = 200  # Camera distance

    # For each view, camera is positioned opposite to the direction we want to see
    view_positions = {
        'front': (-front_dir_world * d).tolist(),
        'back': (front_dir_world * d).tolist(),
        'left': (right_dir_world * d).tolist(),  # Left side seen from +right direction
        'right': (-right_dir_world * d).tolist(),
        'top': (head_dir_world * d).tolist(),
        'bottom': (-head_dir_world * d).tolist()
    }

    # Camera up vector should be the head direction for most views
    # For top/bottom, use front direction as the "up" in the camera view
    view_ups = {
        'front': head_dir_world.tolist(),
        'back': head_dir_world.tolist(),
        'left': head_dir_world.tolist(),
        'right': head_dir_world.tolist(),
        'top': front_dir_world.tolist(),     # Looking down at top of head, face direction is "up"
        'bottom': (-front_dir_world).tolist()
    }

    return {
        'primaryAxis': 0,  # Always axis 0 (head-to-toe) per DICOM convention
        'extents': extents.tolist(),
        'headDirection': head_dir_world.tolist(),
        'frontDirection': front_dir_world.tolist(),
        'rightDirection': right_dir_world.tolist(),
        'facingAway': facing_away,
        'frontBias': float(front_bias),
        'viewPositions': view_positions,
        'viewUps': view_ups
    }


def get_default_orientation():
    """Return default orientation when analysis fails - matches original hardcoded values."""
    d = 200
    return {
        'primaryAxis': 0,
        'extents': [100, 100, 100],
        'headDirection': [1, 0, 0],
        'frontDirection': [0, -0.866, 0.5],
        'rightDirection': [0, 0.5, 0.866],  # Fixed: matches rot_matrix @ [0,0,-1]
        'facingAway': False,
        'frontBias': 0.5,
        'viewPositions': {
            'front': [0, 173, -100],
            'back': [0, -173, 100],
            'left': [0, 100, 173],
            'right': [0, -100, -173],
            'top': [200, 0, 0],
            'bottom': [-200, 0, 0]
        },
        'viewUps': {
            'front': [1, 0, 0],
            'back': [1, 0, 0],
            'left': [1, 0, 0],
            'right': [1, 0, 0],
            'top': [0, -0.866, 0.5],   # Fixed: front_dir_world
            'bottom': [0, 0.866, -0.5]  # Fixed: -front_dir_world
        }
    }


def get_slice_image(volume, axis, index, window_center=400, window_width=2000):
    """Extract a 2D slice and return as base64 PNG."""
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

DIRECTORY_PICKER_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>3D Dental CT Viewer</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            font-family: Arial, sans-serif;
            color: white;
            min-height: 100vh;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { color: #4fc3f7; margin-bottom: 5px; font-size: 24px; }
        .subtitle { color: rgba(255,255,255,0.5); margin-bottom: 20px; font-size: 14px; }
        .browser-section {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .current-path {
            background: rgba(0,0,0,0.3);
            padding: 12px 15px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 13px;
            margin-bottom: 15px;
            word-break: break-all;
            border: 1px solid rgba(79,195,247,0.2);
        }
        .folder-list {
            max-height: 400px;
            overflow-y: auto;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .folder-item {
            padding: 12px 15px;
            cursor: pointer;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            display: flex;
            align-items: center;
            gap: 10px;
            transition: background 0.15s;
        }
        .folder-item:hover { background: rgba(79,195,247,0.15); }
        .folder-item:last-child { border-bottom: none; }
        .folder-item.parent { color: #4fc3f7; }
        .folder-item.has-dcm { color: #6bff6b; }
        .folder-icon { font-size: 18px; }
        .folder-name { flex: 1; font-size: 14px; }
        .dcm-badge {
            background: rgba(107,255,107,0.2);
            color: #6bff6b;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
        }
        .btn-row { display: flex; gap: 10px; }
        .open-btn {
            flex: 1;
            padding: 15px;
            background: #4fc3f7;
            color: #1a1a2e;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            font-size: 16px;
            cursor: pointer;
        }
        .open-btn:hover { background: #81d4fa; }
        .open-btn:disabled { background: #555; color: #888; cursor: not-allowed; }
        .error {
            color: #ff6b6b;
            background: rgba(255,107,107,0.1);
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        .recent-section {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 20px;
        }
        .recent-section h2 { color: #4fc3f7; margin: 0 0 15px 0; font-size: 16px; }
        .recent-list { list-style: none; padding: 0; margin: 0; }
        .recent-list li {
            padding: 10px 12px;
            background: rgba(0,0,0,0.2);
            border-radius: 6px;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 10px;
            transition: background 0.15s;
        }
        .recent-list li:hover { background: rgba(79,195,247,0.15); }
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
        .loading { text-align: center; padding: 40px; color: rgba(255,255,255,0.5); }
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
        <h1>3D Dental CT Viewer</h1>
        <p class="subtitle">Select a folder containing CT images (DICOM, JPG, or PNG)</p>

        {% if error %}
        <p class="error">{{ error }}</p>
        {% endif %}

        {% if recent %}
        <div class="recent-section" style="margin-bottom: 20px;">
            <h2>Recent Folders</h2>
            <ul class="recent-list">
                {% for path in recent %}
                <li>
                    <span class="recent-path" onclick="window.location='/browse?path={{ path | urlencode }}'">{{ path }}</span>
                    <button class="open-recent-btn" onclick="openRecent('{{ path }}')">Open</button>
                </li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        <div class="browser-section">
            <h2 style="color: #4fc3f7; margin: 0 0 15px 0; font-size: 16px;">Browse Folders</h2>
            <div class="current-path">{{ current_path }}</div>
            <div class="folder-list">
                {% if parent_path %}
                <div class="folder-item parent" onclick="window.location='/browse?path={{ parent_path | urlencode }}'">
                    <span class="folder-icon">&#8592;</span>
                    <span class="folder-name">..</span>
                </div>
                {% endif %}
                {% for folder in folders %}
                <div class="folder-item {% if folder.has_dcm %}has-dcm{% endif %}"
                     onclick="window.location='/browse?path={{ folder.path | urlencode }}'">
                    <span class="folder-icon">&#128193;</span>
                    <span class="folder-name">{{ folder.name }}</span>
                    {% if folder.has_dcm %}
                    <span class="dcm-badge">{{ folder.dcm_count }} images</span>
                    {% endif %}
                </div>
                {% endfor %}
                {% if not folders and not parent_path %}
                <div style="padding: 20px; text-align: center; color: rgba(255,255,255,0.4);">No subfolders</div>
                {% endif %}
            </div>
            <div class="btn-row">
                <form id="load-form" action="/load" method="post" style="flex:1; display:flex;">
                    <input type="hidden" name="path" id="load-path" value="{{ current_path }}">
                    <button type="submit" class="open-btn" {% if dcm_count == 0 %}disabled{% endif %}>
                        {% if dcm_count > 0 %}
                        Open This Folder ({{ dcm_count }} images)
                        {% else %}
                        No image files in this folder
                        {% endif %}
                    </button>
                </form>
            </div>
        </div>
    </div>

    <!-- Loading Modal -->
    <div id="loading-modal" class="loading-modal">
        <div class="loading-content">
            <div class="spinner"></div>
            <div class="loading-text">Loading DICOM Images...</div>
            <div class="loading-subtext">Building 3D model, please wait</div>
        </div>
    </div>

    <script>
        // Open a recent folder directly
        function openRecent(path) {
            document.getElementById('loading-modal').classList.add('show');
            document.getElementById('load-path').value = path;
            document.getElementById('load-form').submit();
        }

        // Add loading modal to form submission
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('load-form');
            if (form) {
                form.addEventListener('submit', function(e) {
                    const btn = form.querySelector('button[type="submit"]');
                    if (!btn.disabled) {
                        document.getElementById('loading-modal').classList.add('show');
                    }
                });
            }
        });
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

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>3D Dental CT Viewer</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            overflow: hidden;
            background: #1a1a2e;
            font-family: Arial, sans-serif;
            color: white;
        }
        #container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            height: 100vh;
            gap: 2px;
            background: #0a0a1a;
        }
        .grid-cell {
            position: relative;
            background: #1a1a2e;
            overflow: hidden;
        }
        .cell-header {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            background: rgba(0,0,0,0.7);
            padding: 8px 12px;
            z-index: 10;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .cell-title {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            font-weight: bold;
        }
        .plane-color {
            width: 12px;
            height: 12px;
            border-radius: 2px;
            display: inline-block;
        }
        .current-slice {
            background: #4fc3f7;
            color: #1a1a2e;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 12px;
        }
        .expand-btn {
            padding: 4px 10px;
            background: rgba(79,195,247,0.2);
            border: 1px solid #4fc3f7;
            color: #4fc3f7;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
        }
        .expand-btn:hover { background: rgba(79,195,247,0.4); }
        #viewer3d {
            width: 100%;
            height: 100%;
        }
        #info {
            position: absolute;
            bottom: 60px;
            left: 10px;
            background: rgba(0,0,0,0.85);
            padding: 10px;
            border-radius: 8px;
            z-index: 100;
            max-width: 280px;
            font-size: 11px;
        }
        #info h2 { margin: 0 0 8px 0; color: #4fc3f7; font-size: 14px; }
        .back-btn {
            position: absolute;
            top: 8px;
            right: 10px;
            z-index: 100;
            padding: 6px 12px;
            background: rgba(79,195,247,0.2);
            border: 1px solid #4fc3f7;
            color: #4fc3f7;
            border-radius: 6px;
            cursor: pointer;
            text-decoration: none;
            font-size: 12px;
        }
        .back-btn:hover { background: rgba(79,195,247,0.4); }
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 20px;
            z-index: 200;
        }
        .view-buttons {
            position: absolute;
            top: 10px;
            left: 10px;
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
            z-index: 100;
            max-width: 280px;
        }
        .view-btn {
            padding: 6px 10px;
            background: rgba(79,195,247,0.2);
            border: 1px solid #4fc3f7;
            color: #4fc3f7;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.2s;
        }
        .view-btn:hover { background: rgba(79,195,247,0.4); }
        .view-btn.active {
            background: #4fc3f7;
            color: #1a1a2e;
            font-weight: bold;
        }
        .slice-cell {
            display: flex;
            flex-direction: column;
        }
        .slice-image-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 45px 10px 10px 10px;
            background: black;
        }
        .slice-image-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .slice-controls {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0,0,0,0.8);
            padding: 8px 12px;
            z-index: 10;
        }
        .slider-row {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .slider-row input[type="range"] {
            flex: 1;
        }
        .slider-label {
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: #888;
            margin-top: 2px;
        }
        /* Contrast controls in 3D panel */
        .contrast-controls {
            position: absolute;
            bottom: 10px;
            right: 10px;
            background: rgba(0,0,0,0.85);
            padding: 10px;
            border-radius: 8px;
            z-index: 100;
            font-size: 11px;
        }
        .contrast-controls label {
            display: block;
            margin-bottom: 2px;
        }
        .contrast-controls input[type="range"] {
            width: 120px;
            margin-bottom: 6px;
        }
        /* Expanded view modal */
        .expanded-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.95);
            z-index: 1000;
            padding: 20px;
        }
        .expanded-overlay.show { display: flex; flex-direction: column; }
        .expanded-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .expanded-title {
            font-size: 20px;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .close-btn {
            padding: 8px 16px;
            background: #ff6b6b;
            border: none;
            color: white;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }
        .expanded-content {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .expanded-content img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .expanded-controls {
            margin-top: 15px;
            display: flex;
            align-items: center;
            gap: 20px;
            justify-content: center;
        }
        .expanded-controls input[type="range"] {
            width: 400px;
        }
    </style>
</head>
<body>
    <div id="container">
        <!-- Top-Left: Axial View -->
        <div class="grid-cell slice-cell" id="axial-cell">
            <div class="cell-header">
                <div class="cell-title">
                    <span class="plane-color" style="background: #4fc3f7;"></span>
                    <span>Axial (Top-Down)</span>
                    <span class="current-slice" id="axial-pos">0</span>
                </div>
                <button class="expand-btn" onclick="expandView('axial')">Expand</button>
            </div>
            <div class="slice-image-container">
                <img id="axial-img" src="" alt="Axial slice">
            </div>
            <div class="slice-controls">
                <div class="slider-row">
                    <input type="range" id="axial-slider" min="0" max="100" value="50">
                </div>
                <div class="slider-label">
                    <span>Chin / Bottom</span>
                    <span>Top of Head</span>
                </div>
            </div>
        </div>

        <!-- Top-Right: 3D Model -->
        <div class="grid-cell" id="model-cell">
            <div class="view-buttons" id="view-buttons">
                <button class="view-btn" id="btn-reset" onclick="resetView()">Reset</button>
                <button class="view-btn active" id="btn-front" onclick="setView('front')">Front</button>
                <button class="view-btn" id="btn-back" onclick="setView('back')">Back</button>
                <button class="view-btn" id="btn-left" onclick="setView('left')">Left</button>
                <button class="view-btn" id="btn-right" onclick="setView('right')">Right</button>
                <button class="view-btn" id="btn-top" onclick="setView('top')">Top</button>
            </div>
            <a href="/" class="back-btn">&larr; Change Directory</a>
            <div id="viewer3d">
                <div id="loading">Loading 3D model...</div>
            </div>
            <div id="info">
                <h2>Controls</h2>
                <b>3D:</b> Left-drag: Rotate | Right-drag: Pan | Scroll: Zoom<br>
                <b>Cross-sections:</b> Right-click on model to set slice position
            </div>
            <div class="contrast-controls">
                <label>Brightness: <span id="wc-val">400</span></label>
                <input type="range" id="window-center" min="-500" max="2000" value="400">
                <label>Contrast: <span id="ww-val">2000</span></label>
                <input type="range" id="window-width" min="100" max="4000" value="2000">
            </div>
        </div>

        <!-- Bottom-Left: Sagittal View -->
        <div class="grid-cell slice-cell" id="sagittal-cell">
            <div class="cell-header">
                <div class="cell-title">
                    <span class="plane-color" style="background: #ff6b6b;"></span>
                    <span>Sagittal (Side)</span>
                    <span class="current-slice" id="sagittal-pos">0</span>
                </div>
                <button class="expand-btn" onclick="expandView('sagittal')">Expand</button>
            </div>
            <div class="slice-image-container">
                <img id="sagittal-img" src="" alt="Sagittal slice">
            </div>
            <div class="slice-controls">
                <div class="slider-row">
                    <input type="range" id="sagittal-slider" min="0" max="100" value="50">
                </div>
                <div class="slider-label">
                    <span>Left Side</span>
                    <span>Right Side</span>
                </div>
            </div>
        </div>

        <!-- Bottom-Right: Coronal View -->
        <div class="grid-cell slice-cell" id="coronal-cell">
            <div class="cell-header">
                <div class="cell-title">
                    <span class="plane-color" style="background: #6bff6b;"></span>
                    <span>Coronal (Front)</span>
                    <span class="current-slice" id="coronal-pos">0</span>
                </div>
                <button class="expand-btn" onclick="expandView('coronal')">Expand</button>
            </div>
            <div class="slice-image-container">
                <img id="coronal-img" src="" alt="Coronal slice">
            </div>
            <div class="slice-controls">
                <div class="slider-row">
                    <input type="range" id="coronal-slider" min="0" max="100" value="50">
                </div>
                <div class="slider-label">
                    <span>Back of Head</span>
                    <span>Face / Front</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Expanded View Overlay -->
    <div id="expanded-overlay" class="expanded-overlay">
        <div class="expanded-header">
            <div class="expanded-title">
                <span class="plane-color" id="expanded-color"></span>
                <span id="expanded-title-text">View</span>
                <span class="current-slice" id="expanded-pos">0</span>
            </div>
            <button class="close-btn" onclick="closeExpanded()">Close</button>
        </div>
        <div class="expanded-content">
            <img id="expanded-img" src="" alt="Expanded slice">
        </div>
        <div class="expanded-controls">
            <input type="range" id="expanded-slider" min="0" max="100" value="50">
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

    <script>
        let scene, camera, renderer, controls;
        let boneMesh;
        let volumeShape = null;
        let planeRanges = [50, 50, 50];  // Half-extent for each axis (axial, coronal, sagittal)
        let windowCenter = 400;
        let windowWidth = 2000;
        let raycaster, mouse;
        let axialPlane, sagittalPlane, coronalPlane;
        let planeGroup;  // Group to hold planes with same rotation as mesh

        function init() {
            const modelCell = document.getElementById('model-cell');
            const container = document.getElementById('viewer3d');

            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);

            camera = new THREE.PerspectiveCamera(75, modelCell.clientWidth / modelCell.clientHeight, 0.1, 1000);
            camera.position.set(0, 0, 200);

            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(modelCell.clientWidth, modelCell.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.rotateSpeed = 0.8;
            controls.screenSpacePanning = true;

            scene.add(new THREE.AmbientLight(0x404040, 0.5));

            const light1 = new THREE.DirectionalLight(0xffffff, 0.8);
            light1.position.set(1, 1, 1);
            scene.add(light1);

            const light2 = new THREE.DirectionalLight(0xffffff, 0.5);
            light2.position.set(-1, -1, -1);
            scene.add(light2);

            // Create plane group with same rotation as mesh
            // This ensures planes visually align with the rotated mesh
            planeGroup = new THREE.Group();
            const baseTilt = Math.PI - Math.PI / 6;
            planeGroup.rotation.x = baseTilt;
            scene.add(planeGroup);

            raycaster = new THREE.Raycaster();
            mouse = new THREE.Vector2();

            renderer.domElement.addEventListener('contextmenu', onRightClick);

            loadMesh();
            loadVolumeInfo();
            setupSliders();

            window.addEventListener('resize', onWindowResize);
            animate();
        }

        // View control functions
        // Camera positions are dynamically calculated based on mesh orientation analysis.
        // The server analyzes the mesh geometry (using PCA and bounding box analysis)
        // and DICOM metadata to determine anatomical directions, then provides
        // viewPositions and viewUps for each standard view.

        function highlightButton(viewName) {
            // Remove active class from all buttons
            const buttons = document.querySelectorAll('#view-buttons .view-btn');
            buttons.forEach(btn => btn.classList.remove('active'));
            // Add active class to the selected button
            const activeBtn = document.getElementById('btn-' + viewName);
            if (activeBtn) {
                activeBtn.classList.add('active');
            }
        }

        function resetView() {
            // Reset to front view
            setView('front');
        }

        function setView(view) {
            // Use dynamically calculated view positions from mesh orientation analysis
            // The server analyzes mesh geometry and DICOM metadata to determine
            // correct camera positions for each anatomical view
            controls.target.set(0, 0, 0);

            if (meshOrientation && meshOrientation.viewPositions && meshOrientation.viewPositions[view]) {
                const pos = meshOrientation.viewPositions[view];
                const up = meshOrientation.viewUps[view] || [1, 0, 0];
                camera.position.set(pos[0], pos[1], pos[2]);
                camera.up.set(up[0], up[1], up[2]);
            } else {
                // Fallback positions if orientation data not available
                const d = 200;
                switch(view) {
                    case 'front':
                        camera.position.set(0, 173, -100);
                        camera.up.set(1, 0, 0);
                        break;
                    case 'back':
                        camera.position.set(0, -173, 100);
                        camera.up.set(1, 0, 0);
                        break;
                    case 'left':
                        camera.position.set(0, 100, 173);
                        camera.up.set(1, 0, 0);
                        break;
                    case 'right':
                        camera.position.set(0, -100, -173);
                        camera.up.set(1, 0, 0);
                        break;
                    case 'top':
                        camera.position.set(d, 0, 0);
                        camera.up.set(0, -0.866, 0.5);
                        break;
                    case 'bottom':
                        camera.position.set(-d, 0, 0);
                        camera.up.set(0, 0.866, -0.5);
                        break;
                }
            }
            controls.update();
            highlightButton(view);
        }

        function onRightClick(event) {
            event.preventDefault();

            if (!boneMesh || !volumeShape) return;

            const container = document.getElementById('viewer3d');
            const rect = container.getBoundingClientRect();

            mouse.x = ((event.clientX - rect.left) / container.clientWidth) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / container.clientHeight) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(boneMesh);

            if (intersects.length > 0) {
                const point = intersects[0].point;

                // Map 3D point back to volume indices
                // Account for mesh rotation
                const rotatedPoint = point.clone();
                rotatedPoint.applyAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 6);

                // Use axis-specific ranges for coordinate conversion
                const axialRange = planeRanges[0] * 2;
                const coronalRange = planeRanges[1] * 2;
                const sagittalRange = planeRanges[2] * 2;

                const axialIdx = Math.max(0, Math.min(volumeShape[0] - 1,
                    Math.round((0.5 - rotatedPoint.z / axialRange) * (volumeShape[0] - 1))));
                const coronalIdx = Math.max(0, Math.min(volumeShape[1] - 1,
                    Math.round((0.5 - rotatedPoint.y / coronalRange) * (volumeShape[1] - 1))));
                const sagittalIdx = Math.max(0, Math.min(volumeShape[2] - 1,
                    Math.round((rotatedPoint.x / sagittalRange + 0.5) * (volumeShape[2] - 1))));

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
            // Remove old planes from the group
            if (axialPlane) planeGroup.remove(axialPlane);
            if (sagittalPlane) planeGroup.remove(sagittalPlane);
            if (coronalPlane) planeGroup.remove(coronalPlane);

            if (!volumeShape) return;

            const size = 160;  // Larger planes for better visibility
            // Use axis-specific ranges from server (matches mesh scaling)
            // planeRanges[0] = axial (volume axis 0) = mesh local X
            // planeRanges[1] = coronal (volume axis 1) = mesh local Y
            // planeRanges[2] = sagittal (volume axis 2) = mesh local Z
            const axialRange = planeRanges[0] * 2;
            const coronalRange = planeRanges[1] * 2;
            const sagittalRange = planeRanges[2] * 2;

            // Mesh vertex coordinates from marching_cubes:
            // - verts[:,0] = X = volume axis 0 (axial, head-to-toe)
            // - verts[:,1] = Y = volume axis 1 (coronal, front-to-back)
            // - verts[:,2] = Z = volume axis 2 (sagittal, left-to-right)
            // Planes are added to planeGroup which has the same rotation as the mesh.

            // Axial (blue) - HORIZONTAL slice dividing top/bottom
            // Perpendicular to X (mesh local), moves along X
            const axialGeo = new THREE.PlaneGeometry(size, size);
            const axialMat = new THREE.MeshBasicMaterial({
                color: 0x4fc3f7,
                transparent: true,
                opacity: 0.25,
                side: THREE.DoubleSide
            });
            axialPlane = new THREE.Mesh(axialGeo, axialMat);
            // Rotate to be perpendicular to X axis (in YZ plane)
            axialPlane.rotation.y = Math.PI / 2;
            // Map slider: 0 = bottom, max = top
            const axialPos = (axial / Math.max(volumeShape[0] - 1, 1) - 0.5) * axialRange;
            axialPlane.position.x = axialPos;
            planeGroup.add(axialPlane);

            // Sagittal (red) - VERTICAL slice dividing left/right
            // Perpendicular to Z (mesh local), moves along Z
            const sagGeo = new THREE.PlaneGeometry(size, size);
            const sagMat = new THREE.MeshBasicMaterial({
                color: 0xff6b6b,
                transparent: true,
                opacity: 0.25,
                side: THREE.DoubleSide
            });
            sagittalPlane = new THREE.Mesh(sagGeo, sagMat);
            // Default PlaneGeometry is in XY plane, perpendicular to Z - no rotation needed
            // Map slider: 0 = one side, max = other side
            const sagPos = (sagittal / Math.max(volumeShape[2] - 1, 1) - 0.5) * sagittalRange;
            sagittalPlane.position.z = sagPos;
            planeGroup.add(sagittalPlane);

            // Coronal (green) - VERTICAL slice dividing front/back
            // Perpendicular to Y (mesh local), moves along Y
            const corGeo = new THREE.PlaneGeometry(size, size);
            const corMat = new THREE.MeshBasicMaterial({
                color: 0x6bff6b,
                transparent: true,
                opacity: 0.25,
                side: THREE.DoubleSide
            });
            coronalPlane = new THREE.Mesh(corGeo, corMat);
            // Rotate to be perpendicular to Y axis (in XZ plane)
            coronalPlane.rotation.x = Math.PI / 2;
            // Map slider: 0 = back, max = front
            const corPos = (coronal / Math.max(volumeShape[1] - 1, 1) - 0.5) * coronalRange;
            coronalPlane.position.y = corPos;
            planeGroup.add(coronalPlane);
        }

        let meshOrientation = null;  // Store for camera positioning

        function loadMesh() {
            fetch('/mesh_data')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('loading').style.display = 'none';

                    const geometry = new THREE.BufferGeometry();
                    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(data.vertices), 3));
                    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(data.faces), 1));
                    geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(data.normals), 3));

                    const material = new THREE.MeshPhongMaterial({
                        color: 0xf5f5dc,
                        specular: 0x222222,
                        shininess: 25,
                        side: THREE.DoubleSide
                    });

                    boneMesh = new THREE.Mesh(geometry, material);

                    // Apply rotation to orient the mesh for viewing
                    // Original mesh: X=head-to-toe, Y=front-back, Z=left-right
                    // This tilt positions the mesh so standard views work correctly
                    const baseTilt = Math.PI - Math.PI / 6;  // ~150 degrees
                    boneMesh.rotation.x = baseTilt;

                    scene.add(boneMesh);

                    // Store orientation data for camera positioning
                    meshOrientation = data.orientation || null;
                    console.log('Mesh orientation analysis:', meshOrientation);

                    // Set initial camera position using dynamic orientation
                    if (meshOrientation && meshOrientation.viewPositions) {
                        console.log('Using dynamic view positions from mesh analysis');
                        console.log('  Head direction:', meshOrientation.headDirection);
                        console.log('  Front direction:', meshOrientation.frontDirection);
                        console.log('  Primary axis:', meshOrientation.primaryAxis);
                    }
                    // Always start with front view
                    setView('front');
                })
                .catch(error => {
                    document.getElementById('loading').textContent = 'Error: ' + error;
                });
        }

        function loadVolumeInfo() {
            fetch('/volume_info')
                .then(response => response.json())
                .then(data => {
                    volumeShape = data.shape;
                    // Get plane ranges from server (matches mesh scaling)
                    if (data.plane_ranges) {
                        planeRanges = data.plane_ranges;
                    }

                    document.getElementById('axial-slider').max = volumeShape[0] - 1;
                    document.getElementById('sagittal-slider').max = volumeShape[2] - 1;
                    document.getElementById('coronal-slider').max = volumeShape[1] - 1;

                    const axialMid = Math.floor(volumeShape[0] / 2);
                    const sagMid = Math.floor(volumeShape[2] / 2);
                    const corMid = Math.floor(volumeShape[1] / 2);

                    document.getElementById('axial-slider').value = axialMid;
                    document.getElementById('sagittal-slider').value = sagMid;
                    document.getElementById('coronal-slider').value = corMid;

                    updateAllSlices();
                    updatePlaneHelpers(axialMid, corMid, sagMid);
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
                updateAllSlices();
            });
            document.getElementById('window-width').addEventListener('input', function() {
                windowWidth = parseInt(this.value);
                document.getElementById('ww-val').textContent = windowWidth;
                updateAllSlices();
            });
        }

        function updateSlice(axis, index) {
            document.getElementById(axis + '-pos').textContent = index;
            fetch(`/slice/${axis}/${index}?wc=${windowCenter}&ww=${windowWidth}`)
                .then(response => response.json())
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
            const modelCell = document.getElementById('model-cell');
            camera.aspect = modelCell.clientWidth / modelCell.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(modelCell.clientWidth, modelCell.clientHeight);
        }

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }

        // Expanded view functionality
        let currentExpandedAxis = null;
        const viewInfo = {
            'axial': { color: '#4fc3f7', title: 'Axial (Top-Down)' },
            'sagittal': { color: '#ff6b6b', title: 'Sagittal (Side)' },
            'coronal': { color: '#6bff6b', title: 'Coronal (Front)' }
        };

        function expandView(axis) {
            currentExpandedAxis = axis;
            const overlay = document.getElementById('expanded-overlay');
            const info = viewInfo[axis];

            document.getElementById('expanded-color').style.background = info.color;
            document.getElementById('expanded-title-text').textContent = info.title;

            const slider = document.getElementById(axis + '-slider');
            const expandedSlider = document.getElementById('expanded-slider');
            expandedSlider.max = slider.max;
            expandedSlider.value = slider.value;

            document.getElementById('expanded-pos').textContent = slider.value;
            document.getElementById('expanded-img').src = document.getElementById(axis + '-img').src;

            overlay.classList.add('show');

            expandedSlider.oninput = function() {
                const idx = this.value;
                document.getElementById('expanded-pos').textContent = idx;
                document.getElementById(axis + '-slider').value = idx;
                updateSlice(axis, idx);
                // Update expanded image directly
                fetch(`/slice/${axis}/${idx}?wc=${windowCenter}&ww=${windowWidth}`)
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('expanded-img').src = 'data:image/png;base64,' + data.image;
                    });
                updatePlaneHelpers(
                    parseInt(document.getElementById('axial-slider').value),
                    parseInt(document.getElementById('coronal-slider').value),
                    parseInt(document.getElementById('sagittal-slider').value)
                );
            };
        }

        function closeExpanded() {
            document.getElementById('expanded-overlay').classList.remove('show');
            currentExpandedAxis = null;
        }

        // Close expanded view with Escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && currentExpandedAxis) {
                closeExpanded();
            }
        });

        init();
    </script>
</body>
</html>
'''

def get_folder_info(path):
    """Get list of subfolders and count of image files in a directory."""
    folders = []
    image_count = 0

    try:
        for item in sorted(os.listdir(path)):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if this folder has image files
                try:
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
                    folders.append({
                        'name': item,
                        'path': item_path,
                        'has_dcm': False,
                        'dcm_count': 0
                    })
            elif item.lower().endswith(('.dcm', '.jpg', '.jpeg', '.png')):
                image_count += 1
    except PermissionError:
        pass

    return folders, image_count


@app.route('/')
def index():
    """Redirect to folder browser."""
    return redirect(url_for('browse'))


@app.route('/browse')
def browse():
    """Show folder browser."""
    # Get path from query string, default to user's home or a reasonable start
    path = request.args.get('path', '')

    if not path:
        # Start at the script's parent directory (Dental folder)
        path = str(Path(__file__).parent.parent)

    # Ensure path exists
    if not os.path.isdir(path):
        path = str(Path.home())

    # Get parent path
    parent_path = str(Path(path).parent) if path != '/' else None

    # Get folder contents
    folders, dcm_count = get_folder_info(path)

    # Get recent directories
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
    """Load image files from specified directory."""
    global mesh_data, volume_data, volume_spacing, volume_shape, current_directory

    path = request.form.get('path', '').strip()

    # Helper to render error with all required template variables
    def render_error(error_msg, browse_path=None):
        if not browse_path:
            browse_path = str(Path(__file__).parent.parent)
        parent_path = str(Path(browse_path).parent) if browse_path != '/' else None
        folders, dcm_count = get_folder_info(browse_path)
        return render_template_string(DIRECTORY_PICKER_HTML,
                                      current_path=browse_path,
                                      parent_path=parent_path,
                                      folders=folders,
                                      dcm_count=dcm_count,
                                      recent=load_recent_directories(),
                                      error=error_msg)

    if not path:
        return render_error("Please enter a path")

    if not os.path.isdir(path):
        return render_error(f"Directory not found: {path}")

    # Check for supported image files
    img_type, img_count = detect_image_type(path)
    if img_type is None:
        return render_error("No supported image files found (DICOM, JPG, or PNG)", path)

    try:
        print(f"Loading {img_type.upper()} directory: {path}")
        volume_data, volume_spacing, dicom_orientation = load_volume(path)
        volume_shape = list(volume_data.shape)
        current_directory = path

        print("Creating 3D mesh...")
        # Use threshold=None for auto-detection (important for JPG/PNG)
        verts, faces, normals = create_mesh(volume_data, volume_spacing, threshold=None, keep_n_largest=3)

        # Analyze mesh orientation for auto-positioning (pass DICOM metadata if available)
        print("Analyzing mesh orientation...")
        orientation = analyze_mesh_orientation(verts, dicom_orientation)

        mesh_data = {
            'vertices': verts.flatten().tolist(),
            'faces': faces.flatten().tolist(),
            'normals': normals.flatten().tolist(),
            'orientation': orientation
        }

        # Save to recent directories
        add_recent_directory(path)

        print("Ready!")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_error(f"Error loading: {str(e)}", path)

    return redirect(url_for('viewer'))


@app.route('/viewer')
def viewer():
    """Show the 3D viewer."""
    if volume_data is None:
        return redirect(url_for('index'))
    return render_template_string(HTML_TEMPLATE)


@app.route('/mesh_data')
def get_mesh_data():
    global mesh_data
    if mesh_data is None:
        return jsonify({'error': 'Mesh not loaded'}), 500
    return jsonify(mesh_data)


@app.route('/volume_info')
def get_volume_info():
    global volume_shape, volume_spacing
    if volume_shape is None:
        return jsonify({'error': 'No volume loaded'}), 500

    # Calculate plane ranges based on volume dimensions and spacing
    # These match the scaling used in create_mesh()
    spacing = volume_spacing if volume_spacing else (1, 1, 1)
    vol_extent = [
        volume_shape[0] * spacing[0],
        volume_shape[1] * spacing[1],
        volume_shape[2] * spacing[2]
    ]
    max_vol_dim = max(vol_extent)
    scale_factor = 100 / max_vol_dim if max_vol_dim > 0 else 1

    # Plane ranges: half the extent in each dimension after scaling
    plane_ranges = [
        vol_extent[0] * scale_factor / 2,  # axial (z) half-range
        vol_extent[1] * scale_factor / 2,  # coronal (y) half-range
        vol_extent[2] * scale_factor / 2   # sagittal (x) half-range
    ]

    return jsonify({
        'shape': volume_shape,
        'spacing': list(spacing),
        'plane_ranges': plane_ranges
    })


@app.route('/slice/<axis>/<int:index>')
def get_slice(axis, index):
    global volume_data
    if volume_data is None:
        return jsonify({'error': 'No volume loaded'}), 500
    wc = request.args.get('wc', 400, type=int)
    ww = request.args.get('ww', 2000, type=int)
    image_b64 = get_slice_image(volume_data, axis, index, wc, ww)
    return jsonify({'image': image_b64})


def main():
    print("=" * 60)
    print("3D Dental CT Viewer")
    print("=" * 60)
    print(f"Open http://localhost:{PORT} in your browser")
    print("")
    print("Select a DICOM directory to begin.")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)

    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)


if __name__ == '__main__':
    main()
