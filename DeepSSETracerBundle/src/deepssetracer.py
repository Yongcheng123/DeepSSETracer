import importlib
import torch
import os
import sys
from time import strftime
from time import time
from chimerax.map_data import mrc, ArrayGridData
import numpy as np
torch.set_default_tensor_type(torch.DoubleTensor)


def normalize_batches(density):
    epsilon = 0.00000007
    density = density / np.max(density)
    ave = np.mean(density)
    std = np.std(density) + epsilon
    return (density - ave) / std


def write_mrc(density_array, path, origin, step: (1, 1, 1)):
    grid = ArrayGridData(density_array, origin=origin, step=step)
    mrc.save(grid, path)


def load_batches(args):
    g = mrc.open(args.mrc_path)[0]
    m = g.matrix()
    m = m.T/m.max()
    m[m < 0] = 0
    x_size, y_size, z_size = m.shape

    # xlen = int(g.file_header.get('xlen'))
    # ylen = int(g.file_header.get('ylen'))
    # zlen = int(g.file_header.get('zlen'))
    xlen = x_size
    ylen = y_size
    zlen = z_size

    for x in range(x_size-1, 0, -1):
        if m[x, :, :].max() <= 0.0:
            continue
        else:
            x_size = x
            break
    for y in range(y_size-1, 0, -1):
        if m[:x_size, y, :].max() <= 0:
            continue
        else:
            y_size = y
            break
    for z in range(z_size-1, 0, -1):
        if m[:x_size, :y_size, z].max() <= 0.0:
            continue
        else:
            z_size = z
            break
    print("xOriLen={0}...yOriLen={1}...zOriLen={2}...".format(xlen, ylen, zlen))

    padding = pow(2, args.layers)
    x_padding_size = x_size + padding - x_size % padding
    y_padding_size = y_size + padding - y_size % padding
    z_padding_size = z_size + padding - z_size % padding

    density = np.zeros([1, 1, x_padding_size, y_padding_size, z_padding_size])
    for x in range(0, x_padding_size):
        for y in range(0, y_padding_size):
            for z in range(0, z_padding_size):
                if x <= x_size and y <= y_size and z <= z_size:
                    density[0][0][x][y][z] = torch.tensor(m[x][y][z])
    density = torch.tensor(normalize_batches(density))
    return density, (xlen, ylen, zlen), (x_padding_size, y_padding_size, z_padding_size), g.origin


def segment_labels(array):
    import copy
    helix_array = copy.deepcopy(array)
    sheet_array = copy.deepcopy(array)
    helix_array[helix_array == 2] = 0
    sheet_array[sheet_array == 1] = 0
    sheet_array[sheet_array == 2] = 1
    return helix_array.T, sheet_array.T


def deepssetracer_model(args):
    t_i = time()
    print("%s - Loading model..." % (strftime("%y-%m-%d %H:%M:%S")))
    Model = importlib.import_module(".model.unet", package="chimerax.deepssetracer")
    model = Model.Gem_UNet(args)
    if args.cuda:
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"
    checkpoint = torch.load(
        os.path.dirname(os.path.realpath(__file__)) +
        '{0}torch_best_model.chkpt'.format(os.path.sep), map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if args.cuda:
        model.to("cuda")
    model.eval()

    with torch.no_grad():
        x_images, orig_len, Length, origin = load_batches(args)
        x_axis, y_axis, z_axis = origin
        box_origin = (x_axis+17, y_axis+17, z_axis+17)
        xLength, yLength, zLength = Length
        xlen, ylen, zlen = orig_len
        print("%s - Prediction start..." % (strftime("%y-%m-%d %H:%M:%S")))
        if args.cuda:
            x_images = x_images.cuda()
        out = model(x_images)
        pred = out[:, :, :xLength, :yLength, :zLength]
        pred = pred.transpose(1, 4).transpose(1, 3).transpose(1, 2).reshape(-1, 3)
        _, pred = pred.max(1)
        pred_labels = pred.reshape([xLength, yLength, zLength])
        orig_map_size = np.zeros([xlen, ylen, zlen])
        for x in range(0, xlen):
            for y in range(0, ylen):
                for z in range(0, zlen):
                    if x < xLength and y < yLength and z < zLength:
                        orig_map_size[x, y, z] = pred_labels[x, y, z]
                    else:
                        orig_map_size[x, y, z] = 0
        helix_array, sheet_array = segment_labels(orig_map_size)
        write_mrc(helix_array, args.pred_helix_path, origin, (1, 1, 1))
        write_mrc(sheet_array, args.pred_sheet_path, origin, (1, 1, 1))
        write_mrc(helix_array[17:-17, 17:-17, 17:-17], args.pred_helix_path_NoEdge, box_origin, (1, 1, 1))
        write_mrc(sheet_array[17:-17, 17:-17, 17:-17], args.pred_sheet_path_NoEdge, box_origin, (1, 1, 1))
        sys.stdout.flush()
    t_f = time()
    print("%s - Prediction finished..." % (strftime("%y-%m-%d %H:%M:%S")))
    print(("{:.2f}s total usage...".format(t_f - t_i)))
