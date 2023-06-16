import os
import sys
import torch
import argparse

from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.visualizer import Visualizer
from model.detector import MonoConDetector
from utils.engine_utils import tprint, move_data_device
from dataset.kitti_raw_dataset import KITTIRawDataset

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt


# Arguments
parser = argparse.ArgumentParser('MonoCon Tester for KITTI Raw Dataset')
parser.add_argument('--data_dir',
                    type=str,
                    help="Path where sequence images are saved")
parser.add_argument('--calib_file',
                    type=str,
                    help="Path to calibration file (.txt)")
parser.add_argument('--checkpoint_file', 
                    type=str,
                    help="Path of the checkpoint file (.pth)")
parser.add_argument('--gpu_id', type=int, default=0, help="Index of GPU to use for testing")
parser.add_argument('--fps', type=int, default=25, help="FPS of the result video")
parser.add_argument('--save_dir', 
                    type=str,
                    help="Path of the directory to save the inferenced video")

args = parser.parse_args()



# Main

# (1) Build Dataset
dataset = KITTIRawDataset(args.data_dir, args.calib_file)


# (2) Build Model
device = f'cuda:{args.gpu_id}'
detector = MonoConDetector()
detector.load_checkpoint(args.checkpoint_file)
detector.to(device)
detector.eval()
tprint(f"Checkpoint '{args.checkpoint_file}' is loaded to model.")


# # (3) Build Engine
def build_engine_from_onnx(ONNX_FILE_PATH, ENGINE_PATH ):
    tprint(f"Starting building engine")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    tprint(f"Parsing ONNX..")
    parser = trt.OnnxParser(network, TRT_LOGGER)
    success = parser.parse_from_file(ONNX_FILE_PATH)
    if success:
        tprint(f"ONNX parsed succesfully..")
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 22) 
    tprint(f"Building serialized engine..")
    serialized_engine = builder.build_serialized_network(network, config)
    with open(ENGINE_PATH, "wb") as f:
        f.write(serialized_engine)
    
    
        

with torch.no_grad():
    data = dataset[0]
    data = move_data_device(data, device)
    calib = torch.from_numpy(data['calib'][0].P2).cuda()
    img = data['img']
    viewpad = torch.eye(4)
    viewpad[:calib.shape[0], :calib.shape[1]] = calib
    inv_viewpad = torch.linalg.inv(viewpad).transpose(0, 1).cuda()
    # bboxes_2d, bboxes_3d, labels = detector(img, calib.unsqueeze(0), inv_viewpad.unsqueeze(0))
    tprint(f"Sample inference done with new weights.")
    ONNX_FILE_PATH = "deploy_tools/monocon.onnx"
    ENGINE_PATH = "deploy_tools/monocon.engine"
    input_names = ['image','calib','calib_inv']
    output_names = ['feat']
    # output_names = ['bboxes_2d','bboxes_3d','labels']
    torch.onnx.export(detector, (img, calib.unsqueeze(0), inv_viewpad.unsqueeze(0)),ONNX_FILE_PATH, verbose=True, 
                      input_names=input_names, output_names=output_names, export_params=True)
    tprint(f"ONNX for Monocon generated")
    build_engine_from_onnx(ONNX_FILE_PATH, ENGINE_PATH)
    

    
    

