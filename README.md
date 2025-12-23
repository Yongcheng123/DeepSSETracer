# DeepSSETracer

A deep learning-based tool for automated detection and tracing of secondary structure elements (helices and beta-sheets) in cryo-EM density maps. Built as a ChimeraX plugin with a 3D U-Net architecture optimized for medium-resolution (5-10Å) electron microscopy data.

## Overview

DeepSSETracer uses a custom 3D U-Net to identify alpha-helices and beta-sheets directly from volumetric density maps, eliminating the need for manual structure interpretation. The model processes density maps with arbitrary dimensions through an adaptive tiling strategy and weighted merging scheme.

## Key Features

- **Automatic SSE Detection**: Identifies helices and beta-sheets from cryo-EM maps at 5-10Å resolution
- **Adaptive Processing**: Handles maps of any size through intelligent tiling and merging
- **GPU Acceleration**: Supports CUDA for faster inference (with CPU fallback)
- **ChimeraX Integration**: Seamless workflow within the ChimeraX environment
- **Multi-scale Prediction**: Processes both full maps and cropped regions with edge removal

## Technical Architecture

### Model Design
- **Architecture**: 3-level 3D U-Net with skip connections
- **Input**: Single-channel 3D density maps (normalized)
- **Output**: 3-class segmentation (background, helix, sheet)
- **Features**: 64→128→256 channels with batch normalization and dropout

### Processing Pipeline
1. **Preprocessing**: Automatic voxel spacing resampling to 1Å
2. **Normalization**: Z-score normalization per volume
3. **Adaptive Tiling**: Large maps split into 120³ overlapping tiles
4. **Weighted Merging**: Center-weighted blending to reduce tile artifacts
5. **Postprocessing**: Edge cropping (17 voxels) to remove boundary effects

## Installation

### Requirements
- UCSF ChimeraX (1.0 or higher)
- Python 3.7+
- PyTorch 1.8+
- NumPy

## Usage

### GUI Interface
1. Open ChimeraX and launch **Tools → Volume Data → DeepSSETracer**
2. Select input MRC file containing your density map
3. Choose output directory for predictions
4. Click **Run** to start prediction

### Expected Runtime
- Small maps (74×59×83): ~7s with GPU, ~80s with CPU
- Processing time scales with map volume

### Output Files
- `pre_helix.mrc`: Predicted helix density
- `pre_sheet.mrc`: Predicted sheet density  
- `pre_helix_Cropped.mrc`: Helix prediction with edges removed
- `pre_sheet_Cropped.mrc`: Sheet prediction with edges removed

## Project Structure

```
DeepSSETracerBundle/
├── src/
│   ├── __init__.py              # ChimeraX bundle entry point
│   ├── deepssetracer.py         # Core prediction pipeline
│   ├── tool.py                  # Qt-based GUI interface
│   ├── torch_best_model.chkpt   # Pre-trained model weights
│   └── model/
│       ├── __init__.py
│       └── unet.py              # 3D U-Net architecture
├── bundle_info.xml              # ChimeraX bundle metadata
└── docs/
    └── user/tools/
        └── tutorial.html        # User documentation
```

## Development

### Model Training
The model was trained on annotated cryo-EM density maps with ground truth secondary structure assignments. Training details:
- Loss: Compbined focal cross-entropy and Dice loss
- Optimizer: Adam
- Validation: 5-fold cross-validation

### Key Implementation Details
- **Memory Management**: Automatic tiling prevents OOM errors on large maps
- **Weight Initialization**: Normal distribution (σ=0.1) for stable training
- **Regularization**: Dropout (p=0.5) at pooling and upsampling layers

## Performance Characteristics

**Best Results**: 5-10Å resolution maps with clear secondary structure
**Limitations**: 
- Performance degrades below 4Å or above 12Å
- Requires consistent voxel spacing (automatically resampled)
- May miss heavily distorted secondary structures

## Sample Data

Example inputs and outputs available in `sample_input_output/`:
- `35448_8ihl/`: PDB 8IHL structure and density map
- `cropped_35448/`: Processed outputs showing helix/sheet predictions

## Citation

If you use DeepSSETracer in your research, please cite:

- Yongcheng Mu, et al. "[A Tool for Segmentation of Secondary Structures in 3D Cryo-EM Density Map Components Using Deep Convolutional Neural Networks](https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2021.710119/full)". *Frontiers in Bioinformatics*, 2021. DOI: 10.3389/fbinf.2021.710119

- Yongcheng Mu, et al. "[The Combined Focal Cross Entropy and Dice Loss Function for Segmentation of Protein Secondary Structures from Cryo-EM 3D Density maps](https://ieeexplore.ieee.org/document/9995469)". *2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*. DOI: 10.1109/BIBM55620.2022.9995469

- Yongcheng Mu, et al. ["The Combined Focal Loss and Dice Loss Function Improves the Segmentation of Beta-sheets in Medium-Resolution Cryo-Electron-Microscopy Density Maps"](https://academic.oup.com/bioinformaticsadvances/article/4/1/vbae169/7907198?login=false). *Bioinformatics Advances*, 2024. DOI: 10.1093/bioadv/vbae169

## Contact

For questions or issues, please open a GitHub issue or contact ymu004@odu.edu.

---

**Note**: This tool is designed for research purposes. Always validate predictions against experimental data and known protein structures.
