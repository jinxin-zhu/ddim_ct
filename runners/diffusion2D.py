import os, sys
import logging
import time
import datetime
import glob

import numpy as np
from tqdm.auto import tqdm
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as dist
from torchvision.transforms import ToTensor

from multiprocessing import current_process
# import multiprocessing
# from threading import current_thread
from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
import torchvision.utils as tvu
from tensorboardX import SummaryWriter
# from accelerate import Accelerator

BASE_DIR = '/home/jinxinzhu/project/EfficientSegmentation'
sys.path.append(BASE_DIR)
from BaseSeg.data.dataset_monai import prepare_data
from BaseSeg.losses.get_loss import SegLoss
from BaseSeg.network.get_model import get_denoise_model
from BaseSeg.data.dataset import DataLoaderX, SegDataSet, test_collate_fn

from Common.logger import get_logger


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def is_main_process():
    # process is the main process if its name is 'MainProcess'
    return current_process().name == 'MainProcess'

def get_beta_schedule(beta_schedule, *, beta_start, beta_end,
                      num_diffusion_timesteps):

    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(
            beta_start**0.5,
            beta_end**0.5,
            num_diffusion_timesteps,
            dtype=np.float64,
        )**2)
    elif beta_schedule == "cosine":
        # beta_mid = (beta_end + beta_start) / 2
        # betas = np.cos(np.pi * (sigmoid(np.linspace(
        #     0, beta_mid, num_diffusion_timesteps, dtype=np.float64)
        #     ) - 0.5)) * (beta_end - beta_start) / 2 + beta_start
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        s=0.008
        steps = num_diffusion_timesteps + 1
        x = torch.linspace(0, num_diffusion_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start,
                            beta_end,
                            num_diffusion_timesteps,
                            dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps,
                                  1,
                                  num_diffusion_timesteps,
                                  dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps, )
    return betas


class Diffusion2D(object):

    def __init__(self, args, config, phase='train', device=None):
        self.args = args
        self.cfg = config
        self.phase = phase

        self.train_save_dir = os.path.join(
            self.cfg.training.saver.saver_dir,
            'time-' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') +
            '_resized-' + str(self.cfg.DATA_PREPARE.RESIZED.SPATIAL_SIZE))

        self.test_save_dir = os.path.join(
            self.cfg.testing.saver_dir,
            'time-' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') +
            '_resized-' + str(self.cfg.DATA_PREPARE.RESIZED.SPATIAL_SIZE))

        if device is None:
            device = (torch.device("cuda")
                      if torch.cuda.is_available() else torch.device("cpu"))
        self.device = device
        self.image_folder = None

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) /
                              (1.0 - alphas_cumprod))
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        self.is_distributed_train = config.training.is_distributed_train

        # step 1 >>> init params
        self.epochs = self.cfg.training.n_epochs
        if self.cfg.training.optimizer.lr > self.cfg.DATA_PREPARE.BATCH_SIZE * 1e-3:
            self.lr = self.cfg.training.optimizer.lr * self.cfg.DATA_PREPARE.BATCH_SIZE
        else:
            self.lr = self.cfg.training.optimizer.lr

        self.num_worker = self.cfg.DATA_PREPARE.NUM_WORKER
        if self.cfg.DATA_PREPARE.NUM_WORKER <= self.cfg.DATA_PREPARE.BATCH_SIZE + 2:
            self.num_worker = self.cfg.DATA_PREPARE.BATCH_SIZE + 2

        self.is_apex_train = self.cfg.training.is_apex_train
        self.is_distributed_train = self.cfg.training.is_distributed_train
        if self.phase != 'train':
            self.is_apex_train = False
            self.is_distributed_train = False

        # model = Model(config)
        self.denoise_model = get_denoise_model(self.cfg, self.phase)

        # set distribute training config
        if self.is_distributed_train:
            dist.init_process_group(backend='nccl')
            local_rank = torch.distributed.get_rank()
            print(local_rank)
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)

            self.local_rank = local_rank
            self.is_print_out = True if local_rank == 0 and is_main_process() else False  # Only GPU 0 print information.
            if self.cfg.ENVIRONMENT.CUDA:
                self.denoise_model = self.denoise_model.to(device)
        else:
            self.is_print_out = True
            if self.cfg.ENVIRONMENT.CUDA:
                if self.phase == 'test':
                    if self.cfg.testing.is_fp16:
                        self.denoise_model = self.denoise_model.half()
                    self.denoise_model = self.denoise_model.cuda()
                else:
                    self.denoise_model = self.denoise_model.cuda()

        # init optimizer
        if self.phase == 'train':
            self.optimizer = self._init_optimizer(
            ) if self.phase == 'train' else None

        # load model weight
        self._load_weights()

        # if self.args.resume_training:
        #     states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
        #     model.load_state_dict(states[0])

        #     states[1]["param_groups"][0]["eps"] = self.cfg.optim.eps
        #     optimizer.load_state_dict(states[1])
        #     start_epoch = states[2]
        #     step = states[3]
        #     if self.cfg.model.ema:
        #         ema_helper.load_state_dict(states[4])

        # init logger
        self.log_dir = os.path.join(self.train_save_dir, 'logs') if phase == 'train' else \
            os.path.join(self.test_save_dir, 'logs')
        if self.is_print_out and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.is_print_out:
            self.logger = get_logger(self.log_dir)
            self.logger.info('\n------------ {} options -------------\n')
            self.logger.info('%s' % str(self.cfg))
            self.logger.info('-------------- End ----------------\n')

    def train(self):
        args, config = self.args, self.cfg
        # tb_logger = self.cfg.tb_logger

        # model paramaters set in the beginning
        if self.cfg.model.ema:
            ema_helper = EMAHelper(mu=self.cfg.model.ema_rate)
            ema_helper.register(self.denoise_model)
        else:
            ema_helper = None

        # do train
        if self.is_print_out:
            torch.cuda.synchronize()
            train_start_time = time.time()
            self.logger.info('\nStart training, time: {}'.format(
                time.strftime("%Y.%m.%d %H:%M:%S ",
                              time.localtime(time.time()))))
        self._create_train_data()

        if self.is_print_out:
            self.logger.info('\nPreprocess parallels: {}'.format(
                self.num_worker))
            self.logger.info('\ntrain samples per epoch: {}'.format(
                len(self.train_loader)))
            self.train_writer = SummaryWriter(
                log_dir=os.path.join(self.log_dir, 'tensorboard'))

        if torch.cuda.device_count() > 1:
            if not self.is_distributed_train:
                self.denoise_model = torch.nn.DataParallel(self.denoise_model)
                # self.semi_model = torch.nn.DataParallel(self.semi_model)
            elif self.is_apex_train and self.is_distributed_train:
                self.denoise_model = DistributedDataParallel(
                    self.denoise_model, delay_allreduce=True)
                # self.semi_model = DistributedDataParallel(self.semi_model, delay_allreduce=True)
            elif self.is_distributed_train:
                self.denoise_model = torch.nn.parallel.DistributedDataParallel(
                    self.denoise_model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank)
        #there can put train_dataloader
        _ = next(iter(
            self.train_loader))  # preload, in case some thing goes wrong
        self.train_iter = iter(self.train_loader)

        if self.is_print_out:
            self.logger.info('\nmodel parameters: {}'.format(
                self.count_parameters(self.denoise_model)))

        best_loss = 1e9
        step = 0
        for epoch in range(self.cfg.training.start_epoch,
                           self.cfg.training.n_epochs):
            if self.is_print_out:
                self.logger.info('\nStarting training epoch {}'.format(epoch))
            start_time = datetime.datetime.now()

            # epoch start
            progress_bar = tqdm(total=len(self.train_loader),
                                disable=not is_main_process())
            progress_bar.set_description(f"Epoch {epoch}")

            for i, batch in enumerate(self.train_loader):
                # print(self.is_print_out, step)
                # x = batch['vol']
                x = batch['vol'].to(self.device)
                # Sample noise to add to the images
                # noise = torch.randn(clean_images.shape).to(clean_images.device)
                # bs = clean_images.shape[0]
                # Sample a random timestep for each image
                # timesteps = torch.randint(0, self.num_timesteps, (bs,), device=clean_images.device).long()

                n = x.size(0)
                # data_time += time.time() - data_start
                self.denoise_model.train()
                # step += 1

                # clean_images = clean_images.to(self.device)
                # x = data_transform(self.cfg, clean_images)
                e = torch.randn_like(x).to(x.device)
                b = self.betas

                # antithetic sampling
                # t = torch.randint(
                #     low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                # ).to(self.device)
                t = torch.randint(0,
                                  self.num_timesteps, (n // 2 + 1, ),
                                  device=x.device).long()
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](self.denoise_model, x,
                                                        t, e, b)
                if self.is_print_out and step%100 == 0:
                    self.train_writer.add_scalar(f"loss/{self.phase}", loss.item(), global_step=step)
                # tb_logger.add_scalar("loss", loss, global_step=step)


                # logging.info(
                #     f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                # )

                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg.training.optimizer.is_grad_clip:
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            self.denoise_model.parameters(),
                            config.training.optimizer.grad_clip)
                    except Exception:
                        pass
                self.optimizer.step()

                if self.cfg.model.ema:
                    ema_helper.update(self.denoise_model)

                if step % self.cfg.training.snapshot_freq == 0 or step == 1:
                    states = [
                        self.denoise_model.state_dict(),
                        self.optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.cfg.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path,
                                     "ckpt_{}.pth".format(step)),
                    )
                    # torch.save(states,
                    #            os.path.join(self.args.log_path, "ckpt.pth"))
                if self.is_print_out:
                    if step%100 == 0:
                        progress_bar.update(100)
                        logs = {"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"], "step": step}
                        progress_bar.set_postfix(**logs)
                        self.logger.info(f"Epoch {epoch}" + str(logs))
                        # self.accelerator.log(logs, step=step)
                    step += 1

            if self.is_print_out:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    self._save_weights(epoch, self.denoise_model.state_dict(),
                                       self.optimizer.state_dict())
                    # self.evaluate(epoch)
                # if epoch == self.cfg.training.n_epochs - 1:
                #     self._save_weights(epoch, self.denoise_model.state_dict(),
                #                        self.optimizer.state_dict())
                    # self.evaluate(epoch)
                self.logger.info('\nEnd of epoch {}, epoch used time: {}'.format(
                    epoch,
                    datetime.datetime.now() - start_time))

        if self.is_print_out:
            torch.cuda.synchronize()
            train_end_time = time.time()
            self.logger.info(
                '\nTraining finish,  time: {}'.format(train_end_time))
            self.logger.info('\nTotal training time: {} hours'.format(
                (train_end_time - train_start_time) / 60 / 60))
            self.train_writer.close()

    def evaluate(self, epoch):
        # os.makedirs(os.path.join(self.train_save_dir, "image_samples"),
        #             exist_ok=True)
        self.image_folder = os.path.join(self.train_save_dir, "image_samples",
                                         f"epoch_{epoch}")
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        self.denoise_model.eval()
        if self.cfg.sampling.fid:
            self.sample_fid(self.denoise_model)
        # elif self.cfg.sampling.interpolation:
        #     self.sample_interpolation(self.denoise_model)
        # elif self.cfg.sampling.sequence:
        #     self.sample_sequence(self.denoise_model)
        else:
            raise NotImplementedError("Sample procedeure not defined")
        self.denoise_model.train()

    def sample(self):
        if self.image_folder is None:
            self.image_folder = os.path.join(self.test_save_dir, "image_samples")
            if not os.path.exists(self.image_folder):
                os.makedirs(self.image_folder)

        self.denoise_model.eval()

        if self.args.fid:
            self.sample_fid(self.denoise_model)
        # elif self.args.interpolation:
        #     self.sample_interpolation(model)
        # elif self.args.sequence:
        #     self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")
        self.denoise_model.train()

    def sample_fid(self, model):
        config = self.cfg
        img_id = len(glob.glob(f"{self.image_folder}/*"))
        print(f"starting from image {img_id}")
        print(f"sample image be saved in '{self.image_folder}'.")
        total_n_samples = config.sampling.total_n_samples
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        if not os.path.exists(os.path.join(self.image_folder, "sample_fid")):
            os.makedirs(os.path.join(self.image_folder, "sample_fid"))
        with torch.no_grad():
            for _ in tqdm(range(n_rounds),
                          desc="Generating image samples for FID evaluation."):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.sampling.channels,
                    config.sampling.image_size,
                    config.sampling.image_size,
                    config.sampling.slices,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x_transf = inverse_data_transform(config, x)


                torch.save(x, os.path.join(self.image_folder, "sample_fid", f"x_{img_id}.pt"))
                torch.save(x_transf, os.path.join(self.image_folder, "sample_fid", f"xtransf_{img_id}.pt"))
                img_id += 1

    def sample_sequence(self, model):
        config = self.cfg

        x = torch.randn(
            8,
            config.sampling.channels,
            config.sampling.image_size,
            config.sampling.image_size,
            config.sampling.slices,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j],
                    os.path.join(self.image_folder, f"{j}_{i}.png"))

    def sample_interpolation(self, model):
        config = self.cfg

        def slerp(z1, z2, alpha):
            theta = torch.acos(
                torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1 +
                    torch.sin(alpha * theta) / torch.sin(theta) * z2)

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i:i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i],
                           os.path.join(self.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        
        print("skip:{}".format(skip))

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8),
                                   self.args.timesteps)**2)
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising3D import generalized_steps

            xs = generalized_steps(x,
                                   seq,
                                   model,
                                   self.betas,
                                   eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (np.linspace(0, np.sqrt(self.num_timesteps * 0.8),
                                   self.args.timesteps)**2)
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising3D import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _save_weights(self, 
                      epoch,
                      net_state_dict,
                      optimizer_state_dict,
                      ss_net_state_dict=None,
                      ss_optimizer_state_dict=None):
        if self.is_print_out:
            model_dir = os.path.join(self.train_save_dir, 'models')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            self.best_model_weight_path = os.path.join(
                model_dir, 'best_model.pt')

            torch.save(
                {
                    'epoch': epoch,
                    'state_dict': net_state_dict,
                    'optimizer_dict': optimizer_state_dict,
                    'ss_state_dict': ss_net_state_dict,
                    'ss_optimizer_dict': ss_optimizer_state_dict,
                    'lr': self.lr
                }, self.best_model_weight_path)

    # data prapare
    def _create_train_data(self):
        train_dataset, test_dataset, unlabel_dataset = prepare_data(self.cfg)
        if self.is_distributed_train:
            # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            unlabel_sampler = torch.utils.data.distributed.DistributedSampler(
                unlabel_dataset)
        else:
            # train_sampler = None
            # val_sampler = None
            unlabel_sampler = None

        self.train_loader = DataLoaderX(
            dataset=unlabel_dataset,
            batch_size=self.cfg.DATA_PREPARE.BATCH_SIZE,
            num_workers=self.num_worker,
            shuffle=True if unlabel_sampler is None else False,
            drop_last=False,
            pin_memory=True,
            prefetch_factor=2,
            sampler=unlabel_sampler)

    def _init_optimizer(self):
        if self.cfg.training.optimizer.method.lower() == 'sgd':
            optimizer = optim.SGD(
                self.denoise_model.parameters(),
                lr=self.cfg.training.optimizer.lr,
                momentum=0.99,
                weight_decay=self.cfg.training.optimizer.l2_penalty)
        elif self.cfg.training.optimizer.method.lower() == 'adam':
            optimizer = optim.Adam(
                self.denoise_model.parameters(),
                lr=self.cfg.training.optimizer.lr,
                betas=(0.9, 0.99),
                weight_decay=self.cfg.training.optimizer.l2_penalty,
                amsgrad=self.cfg.training.optimizer.amsgrad,
                eps=self.cfg.training.optimizer.eps)
        return optimizer

    def _load_weights(self):
        if self.phase == 'test':
            self.denoise_model_weight_dir = self.cfg.sampling.denoise_model_path
            # self.fine_model_weight_dir = self.cfg.TESTING.FINE_MODEL_WEIGHT_DIR
            if self.denoise_model_weight_dir is not None and os.path.exists(
                    self.denoise_model_weight_dir):
                checkpoint = torch.load(self.denoise_model_weight_dir)
                self.denoise_model.load_state_dict({
                    k.replace('module.', ''): v
                    for k, v in checkpoint['state_dict'].items()
                })
            else:
                raise Warning('Does not exist the denoise model weight path!')
            # if self.fine_model_weight_dir is not None and os.path.exists(self.fine_model_weight_dir):
            #     checkpoint = torch.load(self.fine_model_weight_dir)
            #     # self.fine_model.load_state_dict(
            #     #     {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
            #     model_dict = self.fine_model.state_dict()
            #     pretrained_dict = checkpoint['state_dict']

            #     # filter out unnecessary keys
            #     pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()
            #                        if k.replace('module.', '') in model_dict}
            #     # overwrite entries in the existing state dict
            #     model_dict.update(pretrained_dict)
            #     # load the new state dict
            #     self.fine_model.load_state_dict(model_dict)
            # else:
            #     raise Warning('Does not exist the fine model weight path!')
        else:
            self.weight_dir = self.cfg.DENOISE_MODEL.WEIGHT_DIR
            if self.weight_dir is not None and os.path.exists(self.weight_dir):
                if self.is_print_out:
                    print('Loading pre_trained model...')
                checkpoint = torch.load(self.weight_dir)
                self.denoise_model.load_state_dict({
                    k.replace('module.', ''): v
                    for k, v in checkpoint['state_dict'].items()
                })
                # self.semi_model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['ss_state_dict'].items()})
                self.start_epoch = checkpoint['epoch']
                self.optimizer.load_state_dict(checkpoint['optimizer_dict'])
                # self.semi_optimizer.load_state_dict(checkpoint['ss_optimizera_dict'])
                self.lr = checkpoint['lr']
            else:
                if self.is_print_out:
                    print('error: Failed to load pre-trained network.')