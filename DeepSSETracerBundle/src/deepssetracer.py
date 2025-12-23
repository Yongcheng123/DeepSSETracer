import importlib
import os
import sys
import copy
from time import strftime, time

import numpy as np
import torch
import torch.nn.functional as F
from chimerax.map_data import mrc, ArrayGridData

torch.set_default_dtype(torch.float32)

        
def get_device():
    """Detect available compute device (CUDA > CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_volume(density):
    """Apply z-score normalization to density data."""
    density = density.cpu().numpy() if isinstance(density, torch.Tensor) else density
    epsilon = 1e-7
    mean = np.mean(density)
    std = np.std(density) + epsilon
    return (density - mean) / std


def write_mrc(density_array, path, origin, step=(1, 1, 1)):
    """Save density array as MRC file."""
    grid = ArrayGridData(density_array, origin=origin, step=step)
    mrc.save(grid, path)


def load_and_pad_density(args):
    """
    Load MRC density map and pad to dimensions divisible by 2^layers.
    
    Returns:
        density: Padded 5D array (batch, channel, x, y, z)
        orig_dims: Original non-zero dimensions
        padded_dims: Padded dimensions
        origin: Map origin coordinates
    """
    g = mrc.open(args.mrc_path)[0]
    m = g.matrix()
    m = m.T / m.max()
    m[m < 0] = 0
    x_size, y_size, z_size = m.shape

    orig_dims = (x_size, y_size, z_size)

    # Find actual data boundaries (trim empty space)
    for x in range(x_size - 1, 0, -1):
        if m[x, :, :].max() > 0:
            x_size = x
            break
            
    for y in range(y_size - 1, 0, -1):
        if m[:x_size, y, :].max() > 0:
            y_size = y
            break
            
    for z in range(z_size - 1, 0, -1):
        if m[:x_size, :y_size, z].max() > 0:
            z_size = z
            break

    print(f"Original dimensions: {orig_dims[0]}×{orig_dims[1]}×{orig_dims[2]}")

    # Pad to multiples of 2^layers for U-Net compatibility
    padding = 2 ** args.layers
    padded_dims = tuple(dim + (padding - dim % padding) for dim in (x_size, y_size, z_size))

    density = np.zeros((1, 1, *padded_dims), dtype=np.float32)
    density[0, 0, :x_size, :y_size, :z_size] = m[:x_size, :y_size, :z_size]

    print(f"Map origin: {g.origin}")
    return density, orig_dims, padded_dims, g.origin

def split_into_classes(predictions):
    """Separate predictions into helix and sheet arrays."""
    helix = copy.deepcopy(predictions)
    sheet = copy.deepcopy(predictions)
    
    helix[helix == 2] = 0  # Remove sheet labels
    sheet[sheet == 1] = 0  # Remove helix labels
    sheet[sheet == 2] = 1  # Normalize sheet labels to 1
    
    return helix.T, sheet.T


def compute_center_weight_map(shape):
    """
    Generate 3D weight map with higher values near the center.
    Used for weighted merging of overlapping tiles to reduce artifacts.
    """
    x_len, y_len, z_len = shape
    center = np.array([x_len, y_len, z_len]) // 2

    # Compute normalized distance from center for each dimension
    dist = []
    for length, c in zip([x_len, y_len, z_len], center):
        if c == 0:
            dist.append(np.zeros(length))
        else:
            coords = np.arange(length)
            dist.append(np.abs(coords - c) / c)

    # Convert distance to weight (1 at center, 0 at edges)
    weights = [1 - d for d in dist]
    
    # Create 3D weight map
    weight_map = np.outer(weights[0], weights[1]).reshape(x_len, y_len, 1) * weights[2]
    return weight_map


def create_overlapping_tiles(density, tile_size=120, min_size=40):
    """
    Split large density maps into overlapping tiles for memory-efficient processing.
    
    Args:
        density: 5D input array (batch, channel, x, y, z)
        tile_size: Target tile dimension (default 120)
        min_size: Minimum acceptable tile size (default 40)
        
    Returns:
        tiles: List of padded tile tensors
        origins: List of (x, y, z) origin coordinates for each tile
        weight_maps: List of weight maps for merging
    """
    _, _, x_size, y_size, z_size = density.shape
    tiles, origins, weight_maps = [], [], []
    
    for x in range(0, x_size, tile_size):
        for y in range(0, y_size, tile_size):
            for z in range(0, z_size, tile_size):
                x_end = min(x + tile_size, x_size)
                y_end = min(y + tile_size, y_size)
                z_end = min(z + tile_size, z_size)

                # Adjust start position to maintain tile_size when possible
                def adjust_start(start, end, size, max_size):
                    if max_size < tile_size:
                        return start, size - end + start
                    shortage = tile_size - (end - start)
                    return max(0, start - shortage), shortage
                
                new_x, x_adjust = adjust_start(x, x_end, x_size, x_size)
                new_y, y_adjust = adjust_start(y, y_end, y_size, y_size)
                new_z, z_adjust = adjust_start(z, z_end, z_size, z_size)

                tile = density[:, :, new_x:x_end, new_y:y_end, new_z:z_end]

                # Pad to multiple of 8 for efficient processing
                pad_x = (8 - (x_end - new_x) % 8) % 8
                pad_y = (8 - (y_end - new_y) % 8) % 8
                pad_z = (8 - (z_end - new_z) % 8) % 8

                tile = F.pad(torch.tensor(tile), (0, pad_z, 0, pad_y, 0, pad_x))
                
                tiles.append(tile)
                origins.append((new_x, new_y, new_z))
                
                actual_shape = (x_end - new_x, y_end - new_y, z_end - new_z)
                weight_maps.append(compute_center_weight_map(actual_shape))
                
    return tiles, origins, weight_maps


def merge_tiles_weighted(prob_maps, origins, weight_maps, target_shape, num_classes=3):
    """
    Merge overlapping tiles using weighted averaging to reduce artifacts.
    
    Args:
        prob_maps: List of probability predictions for each tile
        origins: List of (x, y, z) coordinates for each tile
        weight_maps: List of weight maps for each tile
        target_shape: Desired output shape (x, y, z)
        num_classes: Number of output classes (default 3: background, helix, sheet)
        
    Returns:
        merged_labels: Argmax predictions after weighted merging
    """
    merged_probs = np.zeros((*target_shape, num_classes))
    total_weights = np.zeros(target_shape)

    for prob_map, (x, y, z), weight_map in zip(prob_maps, origins, weight_maps):
        print(f"Processing tile at origin ({x}, {y}, {z}), shape: {prob_map.shape}")
        
        # Determine valid region within target boundaries
        x_end = min(x + prob_map.shape[1], target_shape[0])
        y_end = min(y + prob_map.shape[2], target_shape[1])
        z_end = min(z + prob_map.shape[3], target_shape[2])

        # Trim prob_map and weight_map to fit within boundaries
        prob_map = prob_map[:, :x_end-x, :y_end-y, :z_end-z, :]
        weight_map = weight_map[:x_end-x, :y_end-y, :z_end-z]

        # Accumulate weighted probabilities
        for c in range(num_classes):
            merged_probs[x:x_end, y:y_end, z:z_end, c] += prob_map[0, ..., c] * weight_map
        
        total_weights[x:x_end, y:y_end, z:z_end] += weight_map

    # Normalize by total weights where overlap occurred
    overlap_mask = total_weights > 0
    for c in range(num_classes):
        merged_probs[..., c] = np.divide(
            merged_probs[..., c], 
            total_weights, 
            out=merged_probs[..., c], 
            where=overlap_mask
        )

    merged_labels = np.argmax(merged_probs, axis=-1)
    print(f"Merged prediction shape: {merged_labels.shape}")
    return merged_labels


def predict_secondary_structure(args):
    """
    Main prediction pipeline for SSE detection from cryo-EM density maps.
    
    Loads pre-trained U-Net model, processes input density map (with tiling if necessary),
    and generates helix/sheet predictions saved as MRC files.
    """
    start_time = time()
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"{strftime('%y-%m-%d %H:%M:%S')} - Loading model...")
    Model = importlib.import_module(".model.unet", package="chimerax.deepssetracer")
    model = Model.Gem_UNet(args)
    
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 
        'torch_best_model.chkpt'
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # Load and preprocess density map
    density, orig_dims, padded_dims, origin = load_and_pad_density(args)
    density_tensor = torch.tensor(density, dtype=torch.float32)
    
    # Calculate edge-cropped origin (17 voxel border removal)
    edge_crop = 17
    cropped_origin = tuple(o + edge_crop for o in origin)
    
    print(f"{strftime('%y-%m-%d %H:%M:%S')} - Starting prediction...")

    with torch.no_grad():
        max_dim = max(padded_dims)
        
        if max_dim > 120:
            # Large maps: tile-based processing
            tiles, tile_origins, tile_weights = create_overlapping_tiles(density_tensor)
            prob_maps = []
            
            for tile in tiles:
                tile = torch.tensor(normalize_volume(tile.numpy()))
                tile = tile.to(device)
                
                output = model(tile)
                output = F.softmax(output, dim=1)
                
                # Trim padding
                _, _, x_, y_, z_ = tile.shape
                pred = output[:, :, :x_, :y_, :z_]
                pred = pred.permute(0, 2, 3, 4, 1).cpu().numpy()
                prob_maps.append(pred)
            
            predictions = merge_tiles_weighted(
                prob_maps, tile_origins, tile_weights, orig_dims, num_classes=3
            )
        else:
            # Small maps: single-pass inference
            density_tensor = torch.tensor(normalize_volume(density_tensor.numpy()))
            density_tensor = density_tensor.to(device)
            
            output = model(density_tensor)
            pred = output[:, :, :padded_dims[0], :padded_dims[1], :padded_dims[2]]
            pred = pred.transpose(1, 4).transpose(1, 3).transpose(1, 2).reshape(-1, 3)
            pred = pred.max(1)[1]
            predictions = pred.reshape(padded_dims).cpu().numpy()

        # Trim predictions to original dimensions
        final_predictions = np.zeros(orig_dims)
        final_predictions[:orig_dims[0], :orig_dims[1], :orig_dims[2]] = \
            predictions[:orig_dims[0], :orig_dims[1], :orig_dims[2]]

        # Split into helix and sheet channels
        helix, sheet = split_into_classes(final_predictions)
        
        # Save full and cropped versions
        write_mrc(helix, args.pred_helix_path, origin, (1, 1, 1))
        write_mrc(sheet, args.pred_sheet_path, origin, (1, 1, 1))
        write_mrc(
            helix[edge_crop:-edge_crop, edge_crop:-edge_crop, edge_crop:-edge_crop], 
            args.pred_helix_path_NoEdge, cropped_origin, (1, 1, 1)
        )
        write_mrc(
            sheet[edge_crop:-edge_crop, edge_crop:-edge_crop, edge_crop:-edge_crop], 
            args.pred_sheet_path_NoEdge, cropped_origin, (1, 1, 1)
        )
        
        sys.stdout.flush()

    elapsed = time() - start_time
    print(f"{strftime('%y-%m-%d %H:%M:%S')} - Prediction complete")
    print(f"Total time: {elapsed:.2f}s")
