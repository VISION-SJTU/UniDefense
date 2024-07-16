import os
import sys
import time
import yaml
import torch
import random
import numpy as np

from tqdm import tqdm
from pprint import pprint
from torch.utils import data
import torch.distributed as dist
from torch.cuda.amp import GradScaler

from timm.optim.optim_factory import param_groups_weight_decay as add_wd

from dataset.ocim import OCIMDataset, OCIMSubDataset
from loss import get_loss
from model import load_model
from optimizer import get_optimizer
from scheduler import get_scheduler
from utils.statistic import cal_metrics
from engine import AbstractEngine
from utils.misc import reduce_tensor, center_print
from utils.misc import AccMeter, AverageMeter, Logger, Timer


class OCIMEngine(AbstractEngine):
    path = "engine/ocim_engine.py"
    
    def __init__(self, config, stage="Train"):
        super(OCIMEngine, self).__init__(config, stage)
        # for the sake of re-production
        self.fixed_randomness()

    def _mprint(self, content=""):
        if self.local_rank == 0:
            print(content)

    def _initiated_settings(self, model_cfg=None, data_cfg=None, config_cfg=None):
        self.local_rank = config_cfg["local_rank"]
        self.engine_name = "OCIM"

    def _train_settings(self, model_cfg, data_cfg, config_cfg):
        # debug mode: no log dir, no train_val operation.
        self.debug = config_cfg["debug"]
        self._mprint(f"Using debug mode: {self.debug}.")
        self._mprint("*" * 20)

        # distribution
        dist.init_process_group(config_cfg["distribute"]["backend"])

        # load training dataset
        train_dataset = data_cfg["file"]
        with open(train_dataset, "r") as f:
            options = yaml.load(f, Loader=yaml.FullLoader)
        self.train_set = OCIMDataset(options, split="train")
        self.num_train_domains = self.train_set.num_domains  # number of source domains
        self.train_loaders = list()
        self.train_loaders_len = list()
        self.train_iters = list()
        for _ in self.train_set.datasets:
            # define training sampler
            train_sampler = data.distributed.DistributedSampler(_, rank=self.local_rank)
            train_loader = data.DataLoader(_, shuffle=False, sampler=train_sampler,
                                           num_workers=data_cfg.get("num_workers", 4),
                                           batch_size=data_cfg["train_batch_size"],
                                           drop_last=True)
            # wrapped with data loader
            self.train_loaders.append(train_loader)
            self.train_loaders_len.append(len(train_loader))
            # add placeholders
            self.train_iters.append(None)

        # load validation dataset
        options["test_dataset"] = options.get("test_dataset")[0]
        self.val_set = OCIMSubDataset(options, "test", "both")
        val_sampler = data.distributed.DistributedSampler(self.val_set, rank=self.local_rank, shuffle=False)
        # wrapped with data loader
        self.val_loader = data.DataLoader(self.val_set, shuffle=False, sampler=val_sampler,
                                          num_workers=data_cfg.get("num_workers", 4),
                                          batch_size=data_cfg["val_batch_size"])

        self.train_margin = config_cfg.get("train_margin", (0.0, 0.5))
        self.val_margin = config_cfg.get("val_margin", 0.3)
        self.resume = config_cfg.get("resume", False)

        if not self.debug:
            time_format = "%Y-%m-%d...%H.%M.%S"
            run_id = time.strftime(time_format, time.localtime(time.time()))
            self.run_id = config_cfg.get("id", run_id)
            self.dir = os.path.join("runs", self.model_name, self.run_id)

            if self.local_rank == 0:
                if not self.resume:
                    if os.path.exists(self.dir):
                        raise ValueError("Error: given id '%s' already exists." % self.run_id)
                    os.makedirs(self.dir, exist_ok=True)
                    print(f"Writing config file to file directory: {self.dir}.")
                    self.dataset_config = options
                else:
                    print(f"Resuming the history in file directory: {self.dir}.")

                print(f"Logging directory: {self.dir}.")

                # redirect the std out stream
                sys.stdout = Logger(os.path.join(self.dir, 'records.txt'))
                center_print('Train configurations begins.')
                pprint(self.config)
                pprint(options)
                center_print('Train configurations ends.')

        # total number of steps (or epoch) to train
        self.num_steps = options["num_steps"]
        # the number of steps to write down a log
        self.log_steps = options["log_steps"]
        # the number of steps to validate on val dataset once
        self.val_steps = options["val_steps"]
        # warmup step
        self.warmup_step = config_cfg.get("warmup_step", 0)
        # crop version
        self.crop = config_cfg["crop"]
        self._mprint(f"crop: {self.crop}, "
                     f"train margin: {self.train_margin}, "
                     f"val margin: {self.val_margin}")

        # load model
        self.device = torch.device("cuda:" + str(self.local_rank))
        self.model = load_model(self.model_name)(**model_cfg)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
        self._mprint(f"Using SyncBatchNorm.")
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[self.local_rank], find_unused_parameters=config_cfg.get("find_unused", True))
        self.model_without_ddp = self.model.module

        # load optimizer
        optim_cfg = config_cfg.get("optimizer", None)
        optim_name = optim_cfg.pop("name")
        wd = optim_cfg.pop("weight_decay")
        model_params_groups = add_wd(self.model_without_ddp, wd)
        params_groups = model_params_groups
        self.optimizer = get_optimizer(optim_name)(params_groups, **optim_cfg)
        # load scheduler
        self.scheduler = get_scheduler(self.optimizer, config_cfg.get("scheduler", None))
        # load loss
        self.loss_criterion = {
            'softmax': get_loss("cross_entropy", device=self.device),
            'triplet': get_loss("aw_triplet", device=self.device),
            'kl_div': get_loss("kl_div", device=self.device),
            'fac': get_loss("factorization", device=self.device)
        }

        if self.resume and self.local_rank == 0:
            self._load_ckpt(best=config_cfg.get("resume_best", False), train=True)

    def _test_settings(self, model_cfg, data_cfg, config_cfg):
        # load testing dataset
        test_dataset = data_cfg["file"]
        with open(test_dataset, "r") as f:
            options = yaml.load(f, Loader=yaml.FullLoader)
        options["test_dataset"] = options.get("test_dataset")[0]
        self.test_set = OCIMSubDataset(options, "test", "both")
        # wrapped with data loader
        self.test_loader = data.DataLoader(self.test_set, shuffle=False,
                                           num_workers=data_cfg.get("num_workers", 4),
                                           batch_size=data_cfg["test_batch_size"])
        
        # fetch the logging dir
        self.run_id = config_cfg["id"]
        self.dir = os.path.join("runs", self.model_name, self.run_id)
        assert os.path.exists(self.dir), f"Logging directory '{self.dir}' corrupted."
        print(f"Logging directory: {self.dir}.")
        
        # redirect the std out stream
        sys.stdout = Logger(os.path.join(self.dir, 'test.txt'))
        center_print('Test data configurations begins.')
        pprint(options)
        center_print('Test data configurations ends.')
        
        # margin
        self.test_margin = config_cfg.get("test_margin", 0.3)
        # crop version
        self.crop = config_cfg["crop"]
        self._mprint(f"crop: {self.crop}, test margin: {self.test_margin}")
        
        # load model
        test_ckpt_id = "best_model.bin"
        self.device = torch.device("cuda:" + str(self.local_rank))
        self.model = load_model(self.model_name)(**model_cfg)
        ckpt = torch.load(os.path.join(self.dir, test_ckpt_id), map_location="cpu")
        print(f"Loading checkpoint from {test_ckpt_id}@{self.dir}, "
              f"best step: {ckpt['best_step']}, "
              f"best HTER: {round(ckpt['best_hter'], 4)}, "
              f"best AUC: {round(ckpt['best_auc'], 4)}.")
        self.model.load_state_dict(ckpt["model"])
        self.model = self.model.to(self.device)

    def _load_ckpt(self, best=False, train=False):
        # Not used.
        raise NotImplementedError("The function is not intended to be used here.")

    def _save_ckpt(self, step, best=False):
        save_path = os.path.join(self.dir, "best_model.bin" if best else "latest_model.bin")
        torch.save({
            "step": step,
            "best_step": self.best_step,
            "best_auc": self.best_auc,
            "best_hter": self.best_hter,
            "model": self.model_without_ddp.state_dict(),
        }, save_path)

    def train(self):
        try:
            timer = Timer()
            grad_scalar = GradScaler(2 ** 10)
            train_tracker = range(1, self.num_steps + 1)
            # wrap train generator with tqdm for process 0
            if self.local_rank == 0:
                train_tracker = tqdm(train_tracker, position=0, leave=True)
            train_acc = AccMeter()
            train_loss_trackers = dict()

            for cur_step in train_tracker:
                # activate training mode
                self.model.train()
                self.optimizer.zero_grad()

                # data preparation
                source_images_real = list()
                source_images_fake = list()
                source_labels_real = list()
                source_labels_fake = list()
                source_shape_real = list()
                source_shape_fake = list()
                for domain_idx, domain_data in enumerate(self.train_loaders):
                    # check whether a dataloader has run out
                    if cur_step % self.train_loaders_len[domain_idx] == 1:
                        self.train_loaders[domain_idx].sampler.set_epoch(cur_step)
                        self.train_iters[domain_idx] = iter(self.train_loaders[domain_idx])
                    path, tgt = self.train_iters[domain_idx].next()
                    out = self.train_loaders[domain_idx].dataset.load_item(path, tgt,
                                                                           margin=self.train_margin,
                                                                           crop=self.crop)
                    img = out['images']
                    if domain_idx % 2 == 0:
                        source_images_real.append(img)
                        source_labels_real.append(tgt)
                        source_shape_real.append(img.shape[0])
                    else:
                        source_images_fake.append(img)
                        source_labels_fake.append(tgt)
                        source_shape_fake.append(img.shape[0])
                all_data = torch.cat([*source_images_real, *source_images_fake], dim=0)
                all_tgt = torch.cat([*source_labels_real, *source_labels_fake], dim=0)
                in_data, in_tgt = self.to_device((all_data, all_tgt))

                # warm-up lr
                if self.warmup_step != 0 and cur_step <= self.warmup_step:
                    lr = self.config['config']['optimizer']['lr'] * float(cur_step) / self.warmup_step
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

                sum_real = sum(source_shape_real)
                sum_fake = sum(source_shape_fake)

                out_dict = self.train_unidefense_model(
                    in_data, in_tgt, cur_step, grad_scalar, sum_real, sum_fake)

                cls_out = out_dict['cls_out']

                for key in out_dict.keys():
                    if "loss" in key:
                        if key not in train_loss_trackers:
                            train_loss_trackers[key] = AverageMeter()
                        train_loss_trackers[key].update(reduce_tensor(out_dict[key]).item())

                # calculate metrics
                train_acc.update(cls_out, in_tgt)
                iter_acc = reduce_tensor(train_acc.mean_acc()).item()

                if self.local_rank == 0:
                    if cur_step % self.log_steps == 0:
                        log_info = {
                            "train/acc": iter_acc,
                            "train/lr": self.optimizer.param_groups[0]['lr'],
                        }
                        for key, value in train_loss_trackers.items():
                            log_info.update({f"train/{key}": value.avg})
                        self._log_wandb(log_info, cur_step)
                    # log training step
                    train_tracker.set_description(
                        "Train Iter (%d/%d), Loss %.4f, Triplet %.4f, Spat %.4f, Freq %.4f, ACC %.4f, LR %.6f" % (
                            cur_step, self.num_steps,
                            train_loss_trackers["total_loss"].avg,
                            train_loss_trackers["triplet_loss"].avg,
                            train_loss_trackers["real_rec_loss"].avg if train_loss_trackers.get("real_rec_loss") is not None else 0.,
                            train_loss_trackers["real_freq_loss"].avg if train_loss_trackers.get("real_freq_loss") is not None else 0.,
                            iter_acc, self.optimizer.param_groups[0]['lr'])
                    )
                # validating process
                if cur_step % self.val_steps == 0 and not self.debug:
                    self._mprint()
                    self.validate(cur_step, timer)
            if self.local_rank == 0:
                self._end_wandb()
            dist.destroy_process_group()
        except Exception as e:
            dist.destroy_process_group()
            if self.local_rank == 0:
                raise e

    def validate(self, step, timer):
        v_idx = random.sample(range(1, len(self.val_loader) + 1), k=4)
        categories = self.val_loader.dataset.categories
        self.model.eval()
        prob_dict = dict()
        tgt_dict = dict()
        all_prob_list = [None for _ in range(dist.get_world_size())]
        all_tgt_list = [None for _ in range(dist.get_world_size())]
        sample_rgt = list()
        sample_rest = list()
        sample_pred = list()
        sample_tgt = list()
        with torch.no_grad():
            val_generator = enumerate(self.val_loader, 1)
            if self.local_rank == 0:
                val_generator = tqdm(val_generator, position=0, leave=True)
            for val_idx, val_data in val_generator:
                path, tgt = val_data
                out = self.val_loader.dataset.load_item(path, tgt,
                                                        margin=self.val_margin,
                                                        crop=self.crop)
                img_id = out['path']
                rec_obj = out['images']
                in_data = out['images'].to(self.device)

                results = self.model(in_data)
                cls_out = results["cls_out"]
                rec_feat = results.get("rec")
                if rec_feat is not None:
                    rec_feat = rec_feat.detach().cpu()

                # calculate probability for real images
                prob = torch.softmax(cls_out, dim=1)[:, 0].cpu().data.numpy()
                tgt = tgt.numpy()

                for i in range(len(prob)):
                    video_path = img_id[i].rsplit('/', 1)[0]
                    if video_path in prob_dict.keys():
                        prob_dict[video_path].append(prob[i])
                        tgt_dict[video_path].append(tgt[i])
                    else:
                        prob_dict.update({video_path: [prob[i]]})
                        tgt_dict.update({video_path: [tgt[i]]})

                if val_idx in v_idx and rec_feat is not None:
                    # show images
                    sample_rgt.append(rec_obj[0])
                    sample_rest.append(rec_feat[0])
                    sample_pred.append(cls_out[0])
                    sample_tgt.append(tgt[0])

                if self.local_rank == 0:
                    val_generator.set_description(
                        "Eval (%d/%d), Global Step %d" % (val_idx, len(self.val_loader), step))

        dist.barrier()
        dist.all_gather_object(all_prob_list, prob_dict)
        dist.all_gather_object(all_tgt_list, tgt_dict)

        if self.local_rank == 0:
            # assemble figures
            if step % 1000 == 0 and rec_feat is not None:
                sample_rgt = torch.stack(sample_rgt, dim=0)
                sample_rest = torch.stack(sample_rest, dim=0)
                sample_pred = torch.stack(sample_pred, dim=0)
                figure = self.plot_figure((*sample_rgt, *sample_rest),
                                             ("rgt", "rest"),
                                             sample_pred, sample_tgt,
                                             categories=categories)
            else:
                figure = None

            out_dict = self.gather_eval_output(all_prob_list, all_tgt_list)
            prob_list = out_dict['video_prob']
            tgt_list = out_dict['video_tgt']
            assert len(prob_list) == len(tgt_list), "prob list or tgt list corrupted"
            metrics = cal_metrics(np.array(tgt_list), np.array(prob_list), threshold="auto")
            print(f"Eval Step {step}, EER {metrics['EER']:.4f}, HTER {metrics['ACER']:.4f}, TPR5% {metrics['TPR5%']:.4f}, "
                  f"AUC {metrics['AUC']:.4f}, Thres {metrics['Thre']:.4f}, ACC {metrics['ACC']:.4f}")

            # record the best auc/hter and the corresponding step
            if metrics['AUC'] - metrics['ACER'] > self.best_auc - self.best_hter:
                self.best_auc = metrics['AUC']
                self.best_hter = metrics['ACER']
                self.best_step = step
                self._save_ckpt(step, best=True)
            print("Best Step %d, Best AUC %.4f, Best HTER %.4f, Running Time: %s, Estimated Time: %s" % (
                self.best_step, self.best_auc, self.best_hter,
                timer.measure(), timer.measure(step / self.num_steps)
            ))
            self._save_ckpt(step, best=False)

            val_info = {
                "val/AUC": metrics['AUC'],
                "val/HTER": metrics['ACER'],
                "val/TPR@5%": metrics['TPR5%'],
                "val/best_AUC": self.best_auc,
                "val/best_HTER": self.best_hter
            }
            if figure is not None:
                self._draw_wandb("val/figures", figure, step)
            self._log_wandb(val_info, step)

    def test(self):
        self.model.eval()
        prob_dict = dict()
        tgt_dict = dict()
        with torch.no_grad():
            test_generator = enumerate(self.test_loader, 1)
            test_generator = tqdm(test_generator, position=0, leave=True)
            for test_idx, test_data in test_generator:
                path, tgt = test_data
                out = self.test_loader.dataset.load_item(path, tgt, margin=self.test_margin, crop=self.crop)
                img_id = out['path']
                img = out['images']

                in_data = img.to(self.device)
                results = self.model(in_data)
                cls_out = results["cls_out"]

                # 0 - real; 1 - fake
                prob = torch.softmax(cls_out, dim=1)[:, 0].cpu().data.numpy()
                tgt = tgt.numpy()

                for i in range(len(prob)):
                    video_path = img_id[i].rsplit('/', 1)[0]
                    if video_path in prob_dict.keys():
                        prob_dict[video_path].append(prob[i])
                        tgt_dict[video_path].append(tgt[i])
                    else:
                        prob_dict.update({video_path: [prob[i]]})
                        tgt_dict.update({video_path: [tgt[i]]})

                test_generator.set_description("Test (%d/%d)" % (test_idx, len(self.test_loader)))

            prob_list = list()
            tgt_list = list()
            for key in prob_dict.keys():
                avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
                avg_single_video_tgt = sum(tgt_dict[key]) / len(tgt_dict[key])
                prob_list.append(avg_single_video_prob)
                tgt_list.append(avg_single_video_tgt)
            assert len(prob_list) == len(tgt_list), "prob list or tgt list corrupted"
            metrics = cal_metrics(np.array(tgt_list), np.array(prob_list), threshold="auto")
            print(f"Test | EER {metrics['EER']:.4f}, HTER {metrics['ACER']:.4f}, TPR 5% {metrics['TPR5%']:.4f}, "
                  f"AUC {metrics['AUC']:.4f}, Thres {metrics['Thre']:.8f}, ACC {metrics['ACC']:.4f}\n"
                  f"       APCER {metrics['APCER']:.4f}, BPCER {metrics['BPCER']:.4f}\n"
                  f"       TP_Ratio {metrics['TP_Ratio']:.4f}, #Pos {metrics['NumP']}, "
                  f"TN_Ratio {metrics['TN_Ratio']:.4f}, #Neg {metrics['NumN']}")