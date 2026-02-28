# %%writefile /kaggle/working/mdm/train/training_loop.py
import functools
import os
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_loaders.get_data import get_dataset_loader
INITIAL_LOG_LOSS_SCALE = 20.0
import numpy as np

from sample import generate
import torch.nn as nn



class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        
        self.cond_mode = model.cond_mode
        
        
        # model = nn.DataParallel(model, device_ids=[0,1])
        self.model = model
        
        self.diffusion = diffusion
        
        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        # train
        self.history_dict = {}
        self.epoch_loss_list = []
        self.batch_loss = 0

        # val
        self.history_eval_dict = {}
        self.history_eval_len_dict = {}
        # self.history_dtw_dict = {}
        self.history_fid_dict = {}
        self.epoch_loss_eval_list = []
        self.epoch_loss_eval_len_list = []
        self.batch_eval_loss = 0
        self.batch_eval_len_loss = 0
        # self.epoch_dtw_eval_list = []
        
        self.best_val = 99999999
        
        self.saved_ckpt = []
        
        self.root_save_path = os.path.join(os.getcwd(),self.save_dir)

        self.resume_epoch = 0
        
        self.data = data
        self.val_data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, split='val')
        
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        
        self.step = 0
        self.resume_step = 0
        self.resume_epoch = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.99998)

        
        print(f"ExponentialLR ...")
        
        if self.resume_step:
            self._load_optimizer_state()

        self.device = torch.device(dist_util.dev())
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        if args.eval_during_training:
            from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
            from eval import eval_humanml
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                            split=args.eval_split,
                                            hml_mode='eval')

            self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                                   split=args.eval_split,
                                                   hml_mode='gt')
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
            self.eval_data = {
                'test': lambda: eval_humanml.get_mdm_loader(
                    model, diffusion, args.eval_batch_size,
                    gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
                    args.eval_num_samples, scale=1.,
                )
            }
        self.use_ddp = False
        self.ddp_model = self.model
        self.batch_type = "sentence"
        
    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"Loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                ), strict=False
            )

            model_state_dict = torch.load(resume_checkpoint, map_location='cpu')
            if "history" in model_state_dict:
                self.history_dict = model_state_dict["history"]
                self.resume_epoch = max(self.history_dict.keys()) + 1
                logger.log(f"Resume from epoch : {self.resume_epoch}")
            # if "eval_history" in model_state_dict:
            #     self.history_eval_dict = model_state_dict["eval_history"]
            #     self.best_val = min(self.history_eval_dict.values())
            # if "dtw_history" in model_state_dict:
            #     self.history_dtw_dict = model_state_dict["dtw_history"]
            #     self.best_dtw = min(self.history_dtw_dict.values())
            #     logger.log(f"Best DTW from resume : {self.best_val}")
            # if "fid_history" in model_state_dict:
            #     self.history_fid_dict = model_state_dict["fid_history"]
            #     self.best_fid = min(self.history_fid_dict.values())
            
    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict["opt"])
            
            logger.log(f"loading scheduler state from checkpoint: {opt_checkpoint}")
            self.lr_scheduler.load_state_dict(state_dict["scheduler"])
          
    def adjust_learning_rate(self, optimizer, step, args):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if hasattr(self, 'lr_scheduler'):
            return
        if step < self.lr_anneal_steps:
            lr = args.lr * step / self.lr_anneal_steps
        else:
            lr = args.lr
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr
    
    def run_loop(self):
        for epoch in range(self.resume_epoch, self.num_epochs):
            print(f'Starting epoch {epoch}')
            self.epoch_loss_list.clear()
            self.model.train()
            for motion, cond in tqdm(self.data):

                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

                self.run_step(motion, cond, mode="train")
                
                self.epoch_loss_list.append(self.batch_loss)

                
                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

#                 if self.step % self.save_interval == 0:
                if self.total_step() % self.save_interval == 0 and self.total_step() != 0 or self.total_step() == self.num_steps - 1:
                    # self.dtw = False
                    self.save()
                            
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                    
                self.adjust_learning_rate(self.opt, self.total_step(), self.args)
                self.step += 1

            self.history_dict[epoch] = np.array(self.epoch_loss_list).mean()
            
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
                
            #### RUN EVAL ##################################
            # if epoch % 10 == 0:
            #     self.run_eval()
            #     self.history_eval_dict[epoch] = np.array(self.epoch_loss_eval_list).mean()

                
            #     if self.best_val > self.history_eval_dict[epoch]:
            #         self.best_val = self.history_eval_dict[epoch]
            #         print(f"Hooray!!! New best model with [VAL] = {self.best_val}" )
            #         self.save(best=True)



            # if epoch % 25 == 0:
            #     current_dtw = self.run_dtw(epoch)
            #     self.history_dtw_dict[epoch] = current_dtw
            #     if self.best_val > current_dtw:
            #         self.best_val = current_dtw
            #         print(f"Hooray!!! New best model with [DTW] = {current_dtw}" )
            #         self.save(best=True)
            # if epoch %  == 0:

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def total_step(self):
        return self.step + self.resume_step
    
    def run_dtw(self,epoch):
        print(f'Starting DTW ...')
        self.model.eval()
        val_dtw, val_fid = generate.main(self.args, self.val_data, self.model, self.diffusion, epoch)
        self.model.train()
        return val_dtw
        
    def run_eval(self):
        print(f'Starting EVALUATE ...')
        self.epoch_loss_eval_list.clear()
        self.epoch_loss_eval_len_list.clear()
        self.model.eval()
        for motion, cond in tqdm(self.val_data):
            self.mp_trainer.zero_grad()
            motion = motion.to(self.device)
            cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
            self.run_step(motion, cond, mode="val")
            self.epoch_loss_eval_list.append(self.batch_eval_loss)
            self.epoch_loss_eval_len_list.append(self.batch_eval_len_loss)
        self.model.train()

    def run_step(self, batch, cond, mode="train"):
        self.forward_backward(batch, cond, mode)
        if mode == "train":
            self.mp_trainer.optimize(self.opt)
            self.lr_scheduler.step()
            self.log_step()

    def forward_backward(self, batch, cond, mode="train"):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            self.model
            micro_cond["y"]["mode"] = mode
            # micro_cond["y"]["text_embed"] = self.model.module.encode_text(micro_cond['y']['text'])
            micro_cond["y"]["dtw"] = False
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.data.dataset
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if mode == "train":
                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )

                loss = (losses["loss"] * weights).mean()

                log_loss_dict(
                    self.diffusion, t, {k: v * weights for k, v in losses.items()}
                )

                if torch.isnan(loss):
                    print("NAN DETECTED IN LOSS.")
                    exit()

                self.mp_trainer.backward(loss)
                self.batch_loss = loss.item()
                
            else:
                self.batch_eval_loss = (losses["loss"] * weights).mean().item()
                self.batch_eval_len_loss = 0

                
    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def get_num_step(self, model_name):
        try:
            model_name = model_name.replace("model", "")
            model_name = model_name.replace(".pt", "")
            step = int(model_name)
            return step
        except:
            return -1


    def ckpt_file_name(self, best):
        if best: # Best model
            for old_ckpt in self.saved_ckpt:
                if old_ckpt.startswith("best"):
                    self.saved_ckpt.remove(old_ckpt)
                    os.remove(f"{self.root_save_path}/{old_ckpt}")
                    # opt_ckpt = old_ckpt.replace("model", "opt")
                    # os.remove(f"{self.root_save_path}/{opt_ckpt}")
            ckpt_name = f"best_model{(self.step+self.resume_step):09d}.pt"
        else:
            for old_ckpt in self.saved_ckpt:
                if not old_ckpt.startswith("best"):
                    opt_ckpt = old_ckpt.replace("model", "opt")
                    os.remove(f"{self.root_save_path}/{opt_ckpt}")
                    self.saved_ckpt.remove(old_ckpt)

                    n_step = self.get_num_step(old_ckpt)
                    if n_step != -1 and (n_step % 50000 == 0):
                        continue

                    os.remove(f"{self.root_save_path}/{old_ckpt}")

            ckpt_name = f"model{(self.step+self.resume_step):09d}.pt"
            
        self.saved_ckpt.append(ckpt_name)
        return ckpt_name


    def save(self, best=False):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.') or e.startswith('bert_model.')]
            for e in clip_weights:
                del state_dict[e]
            
            # train history
            state_dict["history"] = self.history_dict
            state_dict["eval_history"] = self.history_eval_dict
            # state_dict["dtw_history"] = self.history_dtw_dict
            # state_dict["fid_history"] = self.history_fid_dict
            # state_dict["len_history"] = self.history_eval_len_dict
            
            
            logger.log(f"Saving model...")
            filename = self.ckpt_file_name(best=best)
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        # if best:
        #     with bf.BlobFile(
        #         bf.join(self.save_dir, f"best_opt{(self.step+self.resume_step):09d}.pt"),
        #         "wb",
        #     ) as f:
        #         opt_state_dict = {"opt": self.opt.state_dict(), \
        #                          "scheduler" : self.lr_scheduler.state_dict()}
                
        #         torch.save(opt_state_dict, f)
        if not best:
            with bf.BlobFile(
                bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
                "wb",
            ) as f:
                opt_state_dict = {"opt": self.opt.state_dict(), \
                                 "scheduler" : self.lr_scheduler.state_dict()}
                
                torch.save(opt_state_dict, f)
                

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()

def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
