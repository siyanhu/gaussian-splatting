#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.loss_utils import ssim
from utils.image_utils import psnr
from utils.loss_utils import l1_loss
from lpipsPyTorch import lpips
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time
import root_file_io as fio

def current_timestamp(micro_second=False):
    t = time.time()
    if micro_second:
        return int(t * 1000 * 1000)
    else:
        return int(t * 1000)

def render_set(source_path, name, iteration, views, gaussians, pipeline, background, log_path, pretrain_tag):

    render_path = os.path.join(source_path, name, pretrain_tag + "_{}".format(iteration), "renders")
    gts_path = os.path.join(source_path, name, pretrain_tag + "_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    psnr_value = 0
    l1_loss_value = 1
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start_time = current_timestamp()
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        strr = os.path.join(render_path, '{0:05d}'.format(idx) + ".png")

        after_time = current_timestamp()
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        ssim_value = ssim(rendering, gt)
        l1_loss_value += l1_loss(rendering, gt).mean().double()
        psnr_value += psnr(rendering, gt).mean().double()
        lpips_value = lpips(rendering, gt, net_type='vgg')
        psnr_log_value = psnr_value
        if idx > 0:
            psnr_log_value = psnr_log_value / idx
        log_str = "\n[INDEX {}] Rendering: L1LOSS {} SSIM {} PSNR {} LPIPS {} TimeElapse {}"\
        .format(idx, l1_loss_value, ssim_value, psnr_log_value, lpips_value.item(), str(after_time - start_time))
        with open(log_path, 'a+') as f:
            f.write(log_str)
    final = "\n[FINAL PSNR {}, L1_loss {}]".format(psnr_value/len(views), l1_loss_value/len(views))
    with open(log_path, 'a+') as f:
        f.write(final)
    

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]

        print("Cuda current device: ", torch.cuda.current_device())
        print("Cuda is avail: ", torch.cuda.is_available())
              
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        pretrain_source = dataset.model_path
        combo = pretrain_source.split('/')
        pretrain_tag = '_'.join(combo[0:2])

        log_path = fio.createPath(fio.sep, [dataset.source_path, "evaluate", pretrain_tag +  "_{}".format(scene.loaded_iter)])
        fio.ensure_dir(log_path)
        log_path = fio.createPath(fio.sep, [log_path], log_name)
        print("Saving log to", log_path)
        render_set(dataset.source_path, "evaluate", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, log_path, pretrain_tag)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path + ' for testing set ' + args.source_path)

    # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    log_name = args.source_path.replace(fio.sep, '_')
    log_name = log_name.replace('output', '')
    log_name = 'render_log_' + log_name + '_' + str(current_timestamp()) + '.txt'

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args))