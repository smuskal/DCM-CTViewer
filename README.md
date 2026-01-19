# DCM-CTViewer

A lightweight, web-based 3D viewer for CT and MRI medical images. View DICOM files and image sequences in your browser with interactive 3D rendering and cross-sectional MPR views.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

## Features

- **3D Volume Rendering** - Interactive 3D mesh reconstruction from CT/MRI data
- **Cross-Sectional Views** - Axial, sagittal, and coronal MPR slices
- **Multiple Format Support** - DICOM (.dcm), JPEG, and PNG image sequences
- **Auto Scan Detection** - Automatically detects dental, bone, soft tissue, and lung scans
- **Adjustable Windowing** - Real-time brightness/contrast controls with presets
- **No Installation Required** - Runs locally in your web browser
- **Privacy First** - All processing happens locally; no data leaves your computer

## Quick Start

### Prerequisites

You need Python 3.9 or later installed on your system.

- **Mac**: Download from [python.org](https://www.python.org/downloads/) or install via Homebrew: `brew install python`
- **Windows**: Download from [python.org](https://www.python.org/downloads/) - **Important**: Check "Add python.exe to PATH" during installation
- **Linux**: Usually pre-installed, or `sudo apt install python3 python3-pip`

### Running the Viewer

**Mac/Linux:**
```bash
# Make the script executable (first time only)
chmod +x start.sh

# Run the viewer
./start.sh
```

Or right-click `start.sh` → Open With → Terminal

**Windows:**

Double-click `start.bat`

The viewer will:
1. Create a virtual environment (first run only)
2. Install required packages (first run only)
3. Start the local web server
4. Open your browser to `http://localhost:7002`

### Using the Viewer

1. **Select a folder** containing your CT/MRI images
   - Folders with images are highlighted in green
   - Recent folders are saved for quick access
2. **Wait for processing** - The 3D model will be generated
3. **Interact with the view**:
   - **Rotate**: Left-click and drag on the 3D model
   - **Zoom**: Scroll wheel
   - **Pan**: Right-click and drag
   - **Set cross-section point**: Right-click on the 3D model
4. **Adjust slices** using the sliders in the right panel
5. **Change contrast** using the preset dropdown or manual sliders

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| DICOM | `.dcm` | Standard medical imaging format |
| JPEG | `.jpg`, `.jpeg` | Exported CT slice sequences |
| PNG | `.png` | Exported CT slice sequences |

For image sequences (JPEG/PNG), files should be numbered sequentially and represent consecutive slices.

## Manual Installation

If you prefer to install manually:

```bash
# Clone the repository
git clone https://github.com/smuskal/DCM-CTViewer.git
cd DCM-CTViewer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the viewer
python ct_viewer.py
```

Then open `http://localhost:7002` in your browser.

## Dependencies

| Package | Purpose |
|---------|---------|
| Flask | Web server framework |
| NumPy | Numerical array processing |
| SciPy | Image processing and analysis |
| scikit-image | Marching cubes algorithm for 3D mesh |
| Pillow | Image format handling |
| pydicom | DICOM file parsing |
| pylibjpeg | JPEG-compressed DICOM support |

## Configuration

The viewer runs on port `7002` by default. To change this, edit the `PORT` variable in `ct_viewer.py`.

## Troubleshooting

### "Python not found" or "python is not recognized"
- **Windows**: Reinstall Python and check "Add python.exe to PATH"
- **Mac/Linux**: Try `python3` instead of `python`

### "Can't connect to localhost:7002"
- Make sure the terminal window running the server is still open
- Check if another application is using port 7002

### Scan not loading
- Ensure the folder contains image files directly (not in subfolders)
- Check that files have correct extensions (.dcm, .jpg, .png)

### 3D model looks wrong
- Try a different preset (Dental, Bone, Soft Tissue, Lung)
- Adjust the contrast/brightness sliders
- Some scans may need threshold adjustments for optimal rendering

## Privacy & Security

- **100% Local Processing**: All image processing happens on your computer
- **No Network Calls**: The viewer works offline after initial setup
- **No Data Collection**: Your medical images are never uploaded anywhere
- **Open Source**: Full source code available for audit

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Three.js](https://threejs.org/) for WebGL 3D rendering
- [pydicom](https://pydicom.github.io/) for DICOM file support
- [scikit-image](https://scikit-image.org/) for the marching cubes algorithm

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Important Medical Disclaimer

**THIS SOFTWARE IS NOT A MEDICAL DEVICE**

DCM-CTViewer is provided for **educational, research, and informational purposes only**.

- This software has **NOT** been reviewed, cleared, or approved by the FDA or any regulatory body
- It is **NOT** intended for clinical diagnosis, treatment planning, or medical decision-making
- The 3D renderings and cross-sectional views may contain artifacts, distortions, or inaccuracies
- **DO NOT** make any healthcare decisions based solely on visualizations from this software
- Always consult qualified healthcare professionals for interpretation of medical images

THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND. THE AUTHORS AND COPYRIGHT HOLDERS SHALL NOT BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM THE USE OF THIS SOFTWARE.

By using this software, you acknowledge that you understand and accept these limitations.

See [DISCLAIMER.md](DISCLAIMER.md) for the full disclaimer.
