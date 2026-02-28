import os
from argparse import Namespace
import re
from os.path import join as pjoin
from data_loaders.humanml.utils.word_vectorizer import POS_enumerator


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading | ', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                parts = line.strip().split(': ', 1)
                if len(parts) != 2:
                    print(f'Warning: skipping malformed opt line: {line.strip()!r}')
                    continue
                key, value = parts
                if value in ('True', 'False'):
                    opt_dict[key] = value == 'True'
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    opt_dict['which_epoch'] = 'latest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196

    elif opt.dataset_name == 'phoenix':
        opt.data_root = './dataset/PHOENIX'
        opt.motion_dir = pjoin(opt.data_root, 'new_joints')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 50
        opt.dim_pose = 150 # openpose : 50 joints, each joints 3 coordinates
        opt.max_motion_length = 500

    elif opt.dataset_name == 'how2sign':
        opt.data_root = './dataset/HOW2SIGN'
        opt.motion_dir = pjoin(opt.data_root, 'new_joints')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 50
        opt.dim_pose = 139 # smplx
        opt.max_motion_length = 500

    elif opt.dataset_name == 'youtube_sign':
        opt.data_root = './dataset/YOUTUBE_SIGN'
        opt.motion_dir = pjoin(opt.data_root, 'new_joints')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 50
        opt.dim_pose = 150  # OpenPose: 50 joints x 3 coordinates
        opt.max_motion_length = 500

    else:
        raise KeyError('Dataset not recognized')

    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    return opt
