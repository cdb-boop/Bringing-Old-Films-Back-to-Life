import sys
import os
import cv2
import importlib
import argparse
import yaml

sys.path.append(os.path.dirname(sys.path[0]))

import torch
from torch.utils.data import DataLoader

from VP_code.data.dataset import Film_dataset_1
from VP_code.utils.util import frame_to_video
from VP_code.utils.data_util import tensor2img
from VP_code.metrics.psnr_ssim import calculate_psnr, calculate_ssim

def load_model(model_name: str, name: str, which_iter: str, debug=False):
    net = importlib.import_module('VP_code.models.' + model_name)
    netG = net.Video_Backbone()
    model_path = os.path.join('OUTPUT', name,'models','net_G_{}.pth'.format(str(which_iter).zfill(5)))
    checkpoint = torch.load(model_path)
    netG.load_state_dict(checkpoint['netG'])
    netG.cuda()
    if debug:
        print("Finish loading model ...")
    return netG

def load_dataset(datasets_val, batch_size=1, debug=False):
    dataset = Film_dataset_1(datasets_val)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, sampler=None)
    if debug:
        print("Finish loading dataset ...")
        print("Test set statistics:")
        print(f'\n\tNumber of test videos: {len(dataset)}')
    return data_loader

def restore(model, frame_batch, temporal_length: int, temporal_stride: int, normalizing: bool):
    all_len = frame_batch['lq'].shape[1]
    all_output = []

    part_output=None
    for i in range(0, all_len, temporal_stride):
        current_part = {}
        current_part['lq'] = frame_batch['lq'][:,i:min(i+temporal_length,all_len),:,:,:]
        current_part['gt'] = frame_batch['gt'][:,i:min(i+temporal_length,all_len),:,:,:]
        current_part['key'] = frame_batch['key']
        current_part['frame_list'] = frame_batch['frame_list'][i:min(i+temporal_length, all_len)]

        part_lq = current_part['lq'].cuda()
        # if part_output is not None:
        #     part_lq[:,:temporal_length-temporal_stride,:,:,:] = part_output[:,temporal_stride-temporal_length:,:,:,:]

        with torch.no_grad():
            part_output = model(part_lq)

        if i == 0:
            all_output.append(part_output.detach().cpu().squeeze(0))
        else:
            restored_temporal_length = min(i+temporal_length, all_len) - i - (temporal_length - temporal_stride)
            all_output.append(part_output[:,0-restored_temporal_length:,:,:,:].detach().cpu().squeeze(0))

        del part_lq

        if i + temporal_length >= all_len:
            break
    #############

    val_output = torch.cat(all_output, dim=0)
    gt = frame_batch['gt'].squeeze(0)
    lq = frame_batch['lq'].squeeze(0)
    if normalizing:
        val_output = (val_output + 1) / 2
        gt = (gt + 1) / 2
        lq = (lq + 1) / 2
    torch.cuda.empty_cache()

    gt_imgs = []
    sr_imgs = []
    for j in range(len(val_output)):
        gt_imgs.append(tensor2img(gt[j]))
        sr_imgs.append(tensor2img(val_output[j]))

    return (gt_imgs, sr_imgs)

def main(opts, config_dict):
    model = load_model(opts.model_name, opts.name, opts.which_iter, debug=True)

    frame_loader: DataLoader = load_dataset(config_dict['datasets']['val'], debug=True)
    if len(frame_loader) == 0:
        raise Exception("No images found!")

    calculate_metric = "metrics" in config_dict["val"]
    if calculate_metric:
        PSNR=0.0
        SSIM=0.0

    model.eval()
    # test_clip_par_folder = config_dict['datasets']['val']['dataroot_lq'].split('/')[-1]

    for frame_batch in frame_loader:  ### Once load all frames
        gt_imgs, sr_imgs = restore(model, frame_batch, opts.temporal_length, opts.temporal_stride, opts.normalizing)

        ### Save the image
        clip_dir_path, frame_name = os.path.split(frame_batch['key'][0])
        frame_name = os.path.splitext(frame_name)[0]
        clip_name = os.path.basename(clip_dir_path)
        test_clip_par_folder = frame_batch['video_name'][0]  ## The video name

        frame_name_list = frame_batch['name_list']

        restored_clip_url = os.path.join(opts.save_place, opts.name, 'test_results_' + str(opts.temporal_length) + "_" + opts.which_iter, test_clip_par_folder, clip_name)
        restored_clip_url = os.path.abspath(restored_clip_url)
        for id, sr_img in enumerate(sr_imgs):
            file_name = os.path.join(restored_clip_url, frame_name_list[id][0])
            dir_name = os.path.abspath(os.path.dirname(file_name))
            os.makedirs(dir_name, exist_ok=True)
            cv2.imwrite(file_name, sr_img)

        ### To Video directly TODO: currently only support 1-depth sub-folder test clip [√]
        video_save_url = restored_clip_url + '.avi'
        frame_to_video(opts.input_video_url, restored_clip_url, video_save_url)

        if calculate_metric:
            PSNR_this_video = [calculate_psnr(sr,gt) for sr,gt in zip(sr_imgs,gt_imgs)]
            SSIM_this_video = [calculate_ssim(sr,gt) for sr,gt in zip(sr_imgs,gt_imgs)]
            PSNR += sum(PSNR_this_video) / len(PSNR_this_video)
            SSIM += sum(SSIM_this_video) / len(SSIM_this_video)

    if calculate_metric:
        PSNR /= len(frame_loader)
        SSIM /= len(frame_loader)

        log_str = f"Validation on {opts.input_video_url}\n"
        log_str += f'\t # PSNR: {PSNR:.4f}\n'
        log_str += f'\t # SSIM: {SSIM:.4f}\n'

        print(log_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str,default='',help='The name of this experiment')
    parser.add_argument('--model_name',type=str,default='',help='The name of adopted model')
    parser.add_argument('--which_iter',type=str,default='latest',help='Load which iteration')
    parser.add_argument('--input_video_url',type=str,default='',help='degraded video input')
    parser.add_argument('--gt_video_url',type=str,default='',help='gt video')
    parser.add_argument('--temporal_length',type=int,default=15,help='How many frames should be processed in one forward')
    parser.add_argument('--temporal_stride',type=int,default=3,help='Stride value while sliding window')
    parser.add_argument('--save_place',type=str,default='./OUTPUT',help='output directory')
    parser.add_argument('--normalizing',type=bool,default=True,help='output directory')

    opts = parser.parse_args()

    with open(os.path.join('./configs',opts.name+'.yaml'), 'r') as stream:
        config_dict = yaml.safe_load(stream)
    config_dict['datasets']['val']['dataroot_gt'] = opts.gt_video_url
    config_dict['datasets']['val']['dataroot_lq'] = opts.input_video_url
    config_dict['datasets']['val']['normalizing'] = opts.normalizing
    config_dict['val']['val_frame_num'] = opts.temporal_length

    main(opts, config_dict)
