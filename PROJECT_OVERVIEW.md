# DeepSSETracer - Quick Project Overview

> This document provides a quick technical overview of the project.

## What is This Project?

A **deep learning tool** for automated detection of protein secondary structures (alpha-helices and beta-sheets) in cryo-electron microscopy density maps. Built as a plugin for UCSF ChimeraX, a professional structural biology visualization tool.

## Technical Stack

| Component | Technology |
|-----------|------------|
| **ML Framework** | PyTorch (3D U-Net architecture) |
| **Domain** | Computational Biology / Cryo-EM |
| **GUI** | PyQt5 (via ChimeraX API) |
| **Data Processing** | NumPy (volumetric arrays) |
| **Application** | ChimeraX Plugin Bundle |

## Key Technical Achievements

### 1. Custom 3D U-Net Architecture
- Implemented encoder-decoder network for volumetric segmentation
- 3-level architecture: 64→128→256 channels
- Skip connections for spatial detail preservation
- Batch normalization and dropout for regularization

**File**: [`DeepSSETracerBundle/src/model/unet.py`](DeepSSETracerBundle/src/model/unet.py)

### 2. Scalable Inference Pipeline
- **Challenge**: Process density maps of arbitrary size (some >1GB)
- **Solution**: Adaptive tiling with weighted merging
- Tiles overlap with center-weighted blending to eliminate artifacts
- Automatic padding for U-Net dimension requirements

**File**: [`DeepSSETracerBundle/src/deepssetracer.py`](DeepSSETracerBundle/src/deepssetracer.py)

### 3. Production-Ready GUI
- Professional Qt-based interface integrated with ChimeraX
- Automatic voxel spacing normalization
- GPU/CPU detection and automatic fallback
- Real-time progress feedback

**File**: [`DeepSSETracerBundle/src/tool.py`](DeepSSETracerBundle/src/tool.py)

## Performance

| Map Size | GPU Time | CPU Time |
|----------|----------|----------|
| 74×59×83 | ~7s | ~80s |
| Larger maps | Scales linearly with volume |

## Scientific Context

**Application**: Automated protein structure determination from cryo-EM data
- Input: 3D electron density maps (5-10Å resolution)
- Output: Predicted locations of helices and sheets
- **Impact**: Accelerates structural biology research by automating tedious manual interpretation

## Project Stats

- **Language**: Python 3.7+
- **Dependencies**: PyTorch, NumPy, ChimeraX
- **Status**: Functional ChimeraX plugin with pre-trained model

## Why This Project Demonstrates ML Skills

1. **Deep Learning**: Custom 3D CNN architecture implementation
2. **Production Engineering**: Memory-efficient inference at scale
3. **Scientific Computing**: Volumetric data processing
4. **GUI Development**: User-friendly interface for domain experts
5. **Domain Expertise**: Understanding of structural biology problem space

## Sample Usage

Please find the manual and installation instructions in the [ManualForDeepSSETracer](ManualForDeepSSETracer.pdf) file.


## Repository Structure

```
DeepSSETracer/
├── README.md                          # Main documentation
├── CONTRIBUTING.md                    # Development guide
├── requirements.txt                   # Dependencies
├── DeepSSETracerBundle/
│   ├── src/
│   │   ├── deepssetracer.py          # Inference pipeline
│   │   ├── tool.py                    # GUI interface
│   │   ├── model/unet.py             # Neural network
│   │   └── torch_best_model.chkpt    # Pre-trained weights
│   └── bundle_info.xml                # ChimeraX metadata
├── sample_input_output/               # Example data
└── compiled_MacOS/                    # Distribution wheels
```

---

**Questions?** Feel free to explore the code or reach out!
