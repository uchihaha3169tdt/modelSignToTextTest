# %%writefile /kaggle/working/mdm/utils/model_util.py
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode



def load_model_wo_clip(model, state_dict):
    missing_keys, _ = model.load_state_dict(state_dict, strict=False)
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args, data):
    if args.arch == "qna_unet":
        from model.mdm_unet import MDM_UNetModel
        model = MDM_UNetModel(**get_model_args(args, data))
    elif args.arch == "sam_stunet":
        from model.sam_stunet import SAM_UNetModel
        model = SAM_UNetModel(**get_model_args(args, data))
    else:
        from model.mdm import MDM
        model = MDM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args, data):

    # default args
    clip_version = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1

    elif args.dataset == 'how2sign': # Sign Language Dataset
        data_rep = 'hml_vec'
        njoints = 139
        nfeats = 1

    elif args.dataset == 'phoenix': # Sign Language Dataset
        data_rep = 'hml_vec'
        njoints = 150
        nfeats = 1

    elif args.dataset == 'youtube_sign': # YouTube Sign Language Dataset
        data_rep = 'hml_vec'
        njoints = 150
        nfeats = 1

    if args.arch == "qna_unet":
        return {'cond_mask_prob': args.cond_mask_prob, 'in_channels': njoints, 'model_channels': 256,
                'out_channels': njoints, 'num_res_blocks': 1, 'channel_mult': "1", 
                'dims': 1, 'use_checkpoint': False, 'num_heads': 4, 'use_scale_shift_norm': True, 
                'dropout': 0.5,'resblock_updown': True, 'use_fp16': True, 'padding': 1,
                'padding_mode': 'zeros'} # sam_stunet base

    elif args.arch == "sam_stunet":
        return {'cond_mask_prob': args.cond_mask_prob, 'in_channels': njoints, 'model_channels': 1024,
                'out_channels': njoints, 'num_res_blocks': 2, 'channel_mult': "1", 
                'dims': 1, 'use_checkpoint': False, 'num_heads': 4, 'use_scale_shift_norm': True, 
                'dropout': 0.5,'resblock_updown': True, 'use_fp16': True, 'padding': 1,
                'padding_mode': 'zeros'} # sam_stunet base

    elif args.arch == "trans_enc":
        nfeats = njoints
        return {'modeltype': '', 'in_channels': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
                'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
                'latent_dim': 512, 'ff_size': 1024, 'num_layers': 8, 'num_heads': 4,
                'dropout': 0.0, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
                'cond_mask_prob': args.cond_mask_prob, 'action_emb': None, 'arch': args.arch,
                'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset}

    else:
        raise ValueError(f'Unsupported architecture [{args.arch}]')


def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
    )
