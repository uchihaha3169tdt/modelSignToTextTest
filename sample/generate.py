
# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import sys

# Ensure the project root is on sys.path when running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.fixseed import fixseed
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
import shutil
from data_loaders.tensors import collate
import math
import numpy as np
import cv2
import os
import torch
# from dtw import dtw
from model.cfg_sampler import ClassifierFreeSampleModel
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm

from numpy import zeros, array, argmin, inf, full
from math import isinf
import numpy as np

def process_sentence(text):
    text = text.lower()
    return text.strip()

# def calculate_fid(act1, act2):
#     mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
#     mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
#     ssdiff = numpy.sum((mu1 - mu2)**2.0)
#     covmean = sqrtm(sigma1.dot(sigma2))
#     if iscomplexobj(covmean):
#         covmean = covmean.real
#     fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
#     return fid

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)
    

from tqdm import tqdm
def main(args, data, model, diffusion, epoch):

    model.eval()  # disable random masking
    
    save_dtw = f"{args.save_dir}/eval_epoch_{epoch}"
    os.makedirs(save_dtw, exist_ok=True)
    
    fixseed(args.seed)
    
    # if args.guidance_param != 1:
    model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
        
    all_dtws = []
    # all_fids = []
    
    # loss_fn = nn.MSELoss()
    
    for k, (ground_truth, model_kwargs) in enumerate(data):

        # add CFG scale to batch
        # if args.guidance_param != 1:
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * 2.5
        
        sample_fn = diffusion.p_sample_loop
        sample = sample_fn(
            model,
            (args.batch_size, model.in_channels, 1, ground_truth.shape[-1]),  # BUG FIX
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            noise=None,
        )

        # Z - NORM
        sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        # sample = sample.cpu().permute(0, 2, 3, 1).float()
        sample = sample.squeeze(1)

        texts = model_kwargs['y']['text']
        
        ground_truth = data.dataset.t2m_dataset.inv_transform(ground_truth.cpu().permute(0, 2, 3, 1)).float()
        # ground_truth = ground_truth.cpu().permute(0, 2, 3, 1).float()
        ground_truth = ground_truth.squeeze(1)
    
        lengths = model_kwargs['y']['lengths'].to("cuda")
        texts = model_kwargs['y']['text']
        model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])

        batch_dtw = []
        # batch_fid = []
        for i, (caption, motion, gt_len, gt_motion) in enumerate(zip(texts, sample, lengths, ground_truth)):
            
            # print(f"Motion fff : {motion.shape}")
            
#             hyp_l = hyp_lens[i]
            hyp_l = gt_len
            ref_l = gt_len
            motion = motion[:hyp_l]
            gt_motion = gt_motion[:ref_l]

            # if i == 0:
            #     # Z- NORM
            #     gt_motion_norm = data.dataset.t2m_dataset.inv_transform(gt_motion).cpu().numpy()
            #     motion_norm = data.dataset.t2m_dataset.inv_transform(motion).cpu().numpy()

            motion = motion.cpu().numpy()
            gt_motion = gt_motion.cpu().numpy()
            _, _, dis_dtw = alter_DTW_timing(motion, gt_motion, transform_pred=False)
            # dis_fid = calculate_fid(motion, gt_motion)
            
            # batch_fid.append(dis_fid)
            batch_dtw.append(dis_dtw)

            if i == 0:
                appr_dtw = round(dis_dtw, 3)
                measure_info = f"[DTW]={appr_dtw}"
                plot_video(joints=motion,
                      file_path=save_dtw,
                      video_name=f"sample_{k}_{i}",
                      references=gt_motion,
                      skip_frames=1,
                      sequence_ID=f"{measure_info}-{caption}")

        m_dtw = np.array(batch_dtw).mean()
        # m_fid = np.array(batch_fid).mean()
        print(f"DTW BATCH | {m_dtw}")
        all_dtws.append(m_dtw)
        
    dtw_final = np.array(all_dtws).mean()
    return dtw_final, 0

    

def done_main():
    split = "test"
    
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    n_frames = max_frames = 500
    fps = 25

    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')[:50]
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')[:50]

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        raw_texts = [args.text_prompt]
        texts = []
        for text in raw_texts:
            texts.append(process_sentence(text))
        args.num_samples = 1
        
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')

    if is_using_data:
        data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=n_frames, split=split)
    else:
        data = load_dataset(args, max_frames, n_frames, split=split)
        
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    args.guidance_param = 5.
    if args.guidance_param != 1:
        print(f"********************************* USE CLASSIFIER with guidance_param={args.guidance_param}")
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    else:
        print(f"********************************* NOOOO guidance_param={args.guidance_param}")
    
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    model.requires_grad_(False)
    all_gt_motions = []

    if is_using_data:
        iterator = iter(data)
        ground_truth, model_kwargs = next(iterator)

        ground_truth = data.dataset.t2m_dataset.inv_transform(ground_truth.cpu().permute(0, 2, 3, 1)).float()
        ground_truth = ground_truth.squeeze(1)

        
        all_gt_motions.append(ground_truth)
        gt_lengths = model_kwargs["y"]["lengths"].cpu().numpy()
    else:
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        is_t2m = any([args.input_text, args.text_prompt])
        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        _, model_kwargs = collate(collate_args)

    all_motions = []
    all_lengths = []
    all_text = []
    
    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
            
        sample_fn = diffusion.p_sample_loop
        sample = sample_fn(
            model,
            (args.batch_size, model.in_channels, 1, n_frames),  # BUG FIX
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            noise=None,
        )

        sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()

#         bs, feat, frames, dim = sample.shape
        sample = sample.squeeze(1)

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text'
            all_text += model_kwargs['y'][text_key]

        all_motions.append(sample.cpu().numpy())

        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]

    try:
      all_gt_motions = np.concatenate(all_gt_motions, axis=0)
      all_gt_motions = all_gt_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    except:
      all_gt_motions = None


    all_text = all_text[:total_num_samples]


    if os.path.exists(out_path):
        shutil.rmtree(out_path)
        
    os.makedirs(out_path)


    print(f"saving visualizations to [{out_path}]...")

    sample_files = []
    num_samples_in_out_file = 7

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)
    

    for sample_i in range(args.num_samples):
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i]
            save_file = sample_file_template.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)

            if all_gt_motions is not None:
                gt_motion = all_gt_motions[sample_i]
                gt_len = gt_lengths[sample_i]
                gt_motion = gt_motion[:gt_len]
                
                _, _, d = alter_DTW_timing(motion, gt_motion, transform_pred=False)
                print(f"DTW of {rep_i} = {d}")

                motion = motion[:gt_len]
                plot_video(joints=motion,
                      file_path=out_path,
                      video_name=save_file,
                      references=gt_motion,
                      skip_frames=1,
                      sequence_ID=caption)

            else:
                plot_video(joints=motion,
                          file_path=out_path,
                          video_name=save_file,
                          references=None,
                          skip_frames=1,
                          sequence_ID=caption)

            print(f"[{caption} | {animation_save_path}]")

    abs_path = os.path.abspath(out_path)

    print(f'[Done] Results are at [{abs_path}]')

def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames, split):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split=split,
                              hml_mode='text_only')
    return data


# Plot a video given a tensor of joints, a file path, video name and references/sequence ID
def plot_video(joints,
               file_path,
               video_name,
               references=None,
               skip_frames=1,
               sequence_ID=None):
    
    # # KKT
    # print("Line 23 in plot_videos.py => Save validation video")
    # ##############

    # Create video template
    FPS = (25 // skip_frames)
    video_file = file_path + "/{}.mp4".format(video_name.split(".")[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if references is None:
        video = cv2.VideoWriter(video_file, fourcc, float(FPS), (650, 650), True)
    elif references is not None:
        video = cv2.VideoWriter(video_file, fourcc, float(FPS), (1300, 650), True)  # Long

    num_frames = 0
    # print(f"JOINT SHAPE : {joints.shape}")

    for (j, frame_joints) in enumerate(joints):

        frame = np.ones((650, 650, 3), np.uint8) * 255


        # Reduce the frame joints down to 2D for visualisation - Frame joints 2d shape is (48,2)
        frame_joints_2d = np.reshape(frame_joints, (50, 3))[:, :2]

        # Draw the frame given 2D joints
        draw_frame_2D(frame, frame_joints_2d)

        sequence_ID_write = "Predicted : " + sequence_ID.split("/")[-1]
        cv2.putText(frame, sequence_ID_write, (180, 600), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 0, 255), 2)
        # cv2.putText(frame, sequence_ID_write, (700, 635), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        #             (0, 0, 0), 2)

        # If reference is provided, create and concatenate on the end
        if references is not None:
            # Extract the reference joints
            ref_joints = references[j]
            # Initialise frame of white
            ref_frame = np.ones((650, 650, 3), np.uint8) * 255

            # Cut off the percent_tok and multiply each joint by 3 (as was reduced in training files)
#             ref_joints = ref_joints[:] * 3

            # Reduce the frame joints down to 2D- Frame joints 2d shape is (48,2)
            ref_joints_2d = np.reshape(ref_joints, (50, 3))[:, :2]

            # Draw these joints on the frame
            draw_frame_2D(ref_frame, ref_joints_2d)

            cv2.putText(ref_frame, "Ground Truth Pose", (190, 600), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2)

            frame = np.concatenate((frame, ref_frame), axis=1)


        # Write the video frame
        video.write(frame)
        num_frames += 1
    # Release the video
    video.release()

# This is the format of the 3D data, outputted from the Inverse Kinematics model
def getSkeletalModelStructure():
    # Definition of skeleton model structure:
    #   The structure is an n-tuple of:
    #
    #   (index of a start point, index of an end point, index of a bone)
    #
    #   E.g., this simple skeletal model
    #
    #             (0)
    #              |
    #              |
    #              0
    #              |
    #              |
    #     (2)--1--(1)--1--(3)
    #      |               |
    #      |               |
    #      2               2
    #      |               |
    #      |               |
    #     (4)             (5)
    #
    #   has this structure:
    #
    #   (
    #     (0, 1, 0),
    #     (1, 2, 1),
    #     (1, 3, 1),
    #     (2, 4, 2),
    #     (3, 5, 2),
    #   )
    #
    #  Warning 1: The structure has to be a tree.
    #  Warning 2: The order isn't random. The order is from a root to lists.
    #

    return (
        # head
        (0, 1, 0),

        # left shoulder
        (1, 2, 1),

        # left arm
        (2, 3, 2),
        # (3, 4, 3),
        # Changed to avoid wrist, go straight to hands
        (3, 29, 3),

        # right shoulder
        (1, 5, 1),

        # right arm
        (5, 6, 2),
        # (6, 7, 3),
        # Changed to avoid wrist, go straight to hands
        (6, 8, 3),

        # left hand - wrist
        # (7, 8, 4),

        # left hand - palm
        (8, 9, 5),
        (8, 13, 9),
        (8, 17, 13),
        (8, 21, 17),
        (8, 25, 21),

        # left hand - 1st finger
        (9, 10, 6),
        (10, 11, 7),
        (11, 12, 8),

        # left hand - 2nd finger
        (13, 14, 10),
        (14, 15, 11),
        (15, 16, 12),

        # left hand - 3rd finger
        (17, 18, 14),
        (18, 19, 15),
        (19, 20, 16),

        # left hand - 4th finger
        (21, 22, 18),
        (22, 23, 19),
        (23, 24, 20),

        # left hand - 5th finger
        (25, 26, 22),
        (26, 27, 23),
        (27, 28, 24),

        # right hand - wrist
        # (4, 29, 4),

        # right hand - palm
        (29, 30, 5),
        (29, 34, 9),
        (29, 38, 13),
        (29, 42, 17),
        (29, 46, 21),

        # right hand - 1st finger
        (30, 31, 6),
        (31, 32, 7),
        (32, 33, 8),

        # right hand - 2nd finger
        (34, 35, 10),
        (35, 36, 11),
        (36, 37, 12),

        # right hand - 3rd finger
        (38, 39, 14),
        (39, 40, 15),
        (40, 41, 16),

        # right hand - 4th finger
        (42, 43, 18),
        (43, 44, 19),
        (44, 45, 20),

        # right hand - 5th finger
        (46, 47, 22),
        (47, 48, 23),
        (48, 49, 24),
    )

# Draw a line between two points, if they are positive points
def draw_line(im, joint1, joint2, c=(0, 0, 255),t=1, width=3):
    thresh = -100
    if joint1[0] > thresh and  joint1[1] > thresh and joint2[0] > thresh and joint2[1] > thresh:

        center = (int((joint1[0] + joint2[0]) / 2), int((joint1[1] + joint2[1]) / 2))

        length = int(math.sqrt(((joint1[0] - joint2[0]) ** 2) + ((joint1[1] - joint2[1]) ** 2))/2)

        angle = math.degrees(math.atan2((joint1[0] - joint2[0]),(joint1[1] - joint2[1])))

        cv2.ellipse(im, center, (width,length), -angle,0.0,360.0, c, -1)

# Draw the frame given 2D joints that are in the Inverse Kinematics format
def draw_frame_2D(frame, joints):
    # Line to be between the stacked
    draw_line(frame, [1, 650], [1, 1], c=(0,0,0), t=1, width=1)
    # Give an offset to center the skeleton around
    offset = [350, 250]

    # Get the skeleton structure details of each bone, and size
    skeleton = getSkeletalModelStructure()
    skeleton = np.array(skeleton)

    number = skeleton.shape[0]

    # Increase the size and position of the joints
    joints = joints * 10 * 12 * 2
    joints = joints + np.ones((50, 2)) * offset

    # Loop through each of the bone structures, and plot the bone
    for j in range(number):

        c = get_bone_colour(skeleton,j)

        draw_line(frame, [joints[skeleton[j, 0]][0], joints[skeleton[j, 0]][1]],
                  [joints[skeleton[j, 1]][0], joints[skeleton[j, 1]][1]], c=c, t=1, width=1)

# get bone colour given index
def get_bone_colour(skeleton,j):
    bone = skeleton[j, 2]

    if bone == 0:  # head
        c = (0, 153, 0)
    elif bone == 1:  # Shoulder
        c = (0, 0, 255)

    elif bone == 2 and skeleton[j, 1] == 3:  # left arm
        c = (0, 102, 204)
    elif bone == 3 and skeleton[j, 0] == 3:  # left lower arm
        c = (0, 204, 204)

    elif bone == 2 and skeleton[j, 1] == 6:  # right arm
        c = (0, 153, 0)
    elif bone == 3 and skeleton[j, 0] == 6:  # right lower arm
        c = (0, 204, 0)

    # Hands
    elif bone in [5, 6, 7, 8]:
        c = (0, 0, 255)
    elif bone in [9, 10, 11, 12]:
        c = (51, 255, 51)
    elif bone in [13, 14, 15, 16]:
        c = (255, 0, 0)
    elif bone in [17, 18, 19, 20]:
        c = (204, 153, 255)
    elif bone in [21, 22, 23, 24]:
        c = (51, 255, 255)
    return c

# Apply DTW
def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path

def avg_frames(frames):
    frames_sum = np.zeros_like(frames[0])
    for frame in frames:
        frames_sum += frame

    avg_frame = frames_sum / len(frames)
    return avg_frame



# Apply DTW to the produced sequence, so it can be visually compared to the reference sequence
def alter_DTW_timing(pred_seq,ref_seq,transform_pred=True):

    # Define a cost function
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))


    # Run DTW on the reference and predicted sequence
    d, cost_matrix, acc_cost_matrix, path = dtw(ref_seq, pred_seq, dist=euclidean_norm)

    # Normalise the dtw cost by sequence length
    d = d / acc_cost_matrix.shape[0]
    
    # Initialise new sequence
    new_pred_seq = np.zeros_like(ref_seq)
    
    
    if transform_pred:
        # j tracks the position in the reference sequence
        j = 0
        skips = 0
        squeeze_frames = []
        for (i, pred_num) in enumerate(path[0]):

            if i == len(path[0]) - 1:
                break

            if path[1][i] == path[1][i + 1]:
                skips += 1

            # If a double coming up
            if path[0][i] == path[0][i + 1]:
                squeeze_frames.append(pred_seq[i - skips])
                j += 1
            # Just finished a double
            elif path[0][i] == path[0][i - 1]:
                new_pred_seq[pred_num] = avg_frames(squeeze_frames)
                squeeze_frames = []
            else:
                new_pred_seq[pred_num] = pred_seq[i - skips]

    return new_pred_seq, ref_seq, d



if __name__ == "__main__":
    done_main()
