import os
import sys
import time
import yaml
import torch
import numpy as np

from tqdm import tqdm
from pprint import pprint
from torch.utils import data
import torch.distributed as dist
from torch.cuda.amp import GradScaler

from timm.optim.optim_factory import param_groups_weight_decay as add_wd

from dataset.uniattack import UniAttack
from loss import get_loss
from model import load_model
from optimizer import get_optimizer
from scheduler import get_scheduler
from utils.statistic import cal_metrics
from engine import AbstractEngine
from utils.misc import reduce_tensor, center_print
from utils.misc import AccMeter, AverageMeter, Logger, Timer


class UniAttackEngine(AbstractEngine):
    path = "engine/uniattack_engine.py"
    
    def __init__(self, config, stage="Train"):
        super(UniAttackEngine, self).__init__(config, stage)
        # for the sake of re-production
        self.fixed_randomness()

    def _mprint(self, content=""):
        if self.local_rank == 0:
            print(content)

    def _initiated_settings(self, model_cfg=None, data_cfg=None, config_cfg=None):
        self.local_rank = config_cfg["local_rank"]
        self.engine_name = "UniAttack"
        
    @staticmethod
    def _prepare_domain_label_map(options):
        
        real_set = set()
        for i in options["train_real_method"]:
            real_set.add(i.split("-")[0])
        fake_set = set()
        for i in options["train_fake_method"]:
            fake_set.add(i.split("-")[0])
        # FIXME Needs more strict check
        assert len(real_set) == len(fake_set), f"real domain: {real_set}, fake domain: {fake_set}."
        
        domain_label_map = dict()
        domain_list = sorted(list(real_set))
        for i, d in enumerate(domain_list):
            domain_label_map.update({options[f"{d}_root"]: i})
        
        return domain_label_map

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
        self.train_real_set = UniAttack(options, split="train", methods=options["train_real_method"])
        train_real_sampler = data.distributed.DistributedSampler(self.train_real_set,
                                                                 rank=self.local_rank,
                                                                 shuffle=True)
        # wrapped with data loader
        self.train_real_loader = data.DataLoader(self.train_real_set, shuffle=False, sampler=train_real_sampler,
                                                 num_workers=data_cfg.get("num_workers", 8),
                                                 batch_size=data_cfg["train_batch_size"],
                                                 drop_last=True)
        self.train_fake_set = UniAttack(options, split="train", methods=options["train_fake_method"])
        train_fake_sampler = data.distributed.DistributedSampler(self.train_fake_set,
                                                                 rank=self.local_rank,
                                                                 shuffle=True)
        # wrapped with data loader
        self.train_fake_loader = data.DataLoader(self.train_fake_set, shuffle=False, sampler=train_fake_sampler,
                                                 num_workers=data_cfg.get("num_workers", 8),
                                                 batch_size=data_cfg["train_batch_size"],
                                                 drop_last=True)

        # load validation dataset
        self.val_real_set = UniAttack(options, split="val", methods=options["val_real_method"])
        val_real_sampler = data.distributed.DistributedSampler(self.val_real_set, rank=self.local_rank, shuffle=False)
        # wrapped with data loader
        self.val_real_loader = data.DataLoader(self.val_real_set, shuffle=False, sampler=val_real_sampler,
                                               num_workers=data_cfg.get("num_workers", 4),
                                               batch_size=data_cfg["val_batch_size"])
        self.val_fake_set = UniAttack(options, split="val", methods=options["val_fake_method"])
        val_fake_sampler = data.distributed.DistributedSampler(self.val_fake_set, rank=self.local_rank, shuffle=False)
        # wrapped with data loader
        self.val_fake_loader = data.DataLoader(self.val_fake_set, shuffle=False, sampler=val_fake_sampler,
                                               num_workers=data_cfg.get("num_workers", 4),
                                               batch_size=data_cfg["val_batch_size"])

        # load test dataset
        self.test_set = UniAttack(options, split="test", methods=options["test_method"])
        test_sampler = data.distributed.DistributedSampler(self.test_set, rank=self.local_rank, shuffle=False)
        # wrapped with dataloader
        self.test_loader = data.DataLoader(self.test_set, shuffle=False, sampler=test_sampler,
                                           batch_size=data_cfg["test_batch_size"])

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
        # whether to use domain label
        if config_cfg.get("use_domain_label", False):
            self.dlabel_map = self._prepare_domain_label_map(options)
        else:
            self.dlabel_map = None
        # warmup step
        self.warmup_step = config_cfg.get("warmup_step", 0)
        # margin
        self.margin = config_cfg.get("margin", None)
        # crop version
        self.crop = config_cfg["crop"]
        self._mprint(f"crop: {self.crop}, margin: {self.margin}.")
        self._mprint(f"dlabel map: {self.dlabel_map}")
        
        # init metrics
        self.best_auc_frmae = 0.
        self.best_auc_video = 0.
        self.best_hter_frame = 1.0e8
        self.best_hter_video = 1.0e8

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
        self.optimizer = get_optimizer(optim_name)(model_params_groups, **optim_cfg)
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
        
        # load validation dataset
        self.val_real_set = UniAttack(options, split="val", methods=options["val_real_method"])
        # wrapped with data loader
        self.val_real_loader = data.DataLoader(self.val_real_set, shuffle=False,
                                               num_workers=data_cfg.get("num_workers", 4),
                                               batch_size=data_cfg["test_batch_size"])
        self.val_fake_set = UniAttack(options, split="val", methods=options["val_fake_method"])
        self.val_fake_loader = data.DataLoader(self.val_fake_set, shuffle=False,
                                               num_workers=data_cfg.get("num_workers", 4),
                                               batch_size=data_cfg["test_batch_size"])
        
        # load test dataset
        self.test_set = UniAttack(options, split="test", methods=options["test_method"])
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
        self.margin = config_cfg.get("margin", None)
        # crop version
        self.crop = config_cfg["crop"]
        print(f"crop: {self.crop}, margin: {self.margin}.")

        # load model
        test_ckpt_id = "best_model.bin"
        self.device = torch.device("cuda:" + str(self.local_rank))
        self.model = load_model(self.model_name)(**model_cfg)
        ckpt = torch.load(os.path.join(self.dir, test_ckpt_id), map_location="cpu")
        print(
            f"Loading checkpoint from {test_ckpt_id}@{self.dir}, "
            f"best step: {ckpt['best_step']}.\n"
            f"\t[Video] Best ACER: {round(ckpt['best_hter_video'], 4)}"
            f"\tBest AUC: {round(ckpt['best_auc_video'], 4)}.\n"
            f"\t[Frame] Best ACER: {round(ckpt['best_hter'], 4)}"
            f"\tBest AUC: {round(ckpt['best_auc'], 4)}."
        )
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
            "best_auc": self.best_auc_frame,
            "best_auc_video": self.best_auc_video,
            "best_hter": self.best_hter_frame,
            "best_hter_video": self.best_hter_video,
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

            # data preparation
            train_iters = [None, None]
            for cur_step in train_tracker:
                # activate training mode
                self.model.train()
                self.optimizer.zero_grad()

                if cur_step % len(self.train_real_loader) == 1:
                    self.train_real_loader.sampler.set_epoch(cur_step)
                    train_iters[0] = iter(self.train_real_loader)
                if cur_step % len(self.train_fake_loader) == 1:
                    self.train_fake_loader.sampler.set_epoch(cur_step)
                    train_iters[1] = iter(self.train_fake_loader)

                # real
                path, tgt = train_iters[0].next()
                out = self.train_real_loader.dataset.load_item(path, tgt,
                                                               margin=self.margin, crop=self.crop,
                                                               dataset_label_map=self.dlabel_map)
                images_real = out['images']
                domain_labels_real = out['dataset_labels']
                labels_real = tgt
                # fake
                path, tgt = train_iters[1].next()
                out = self.train_fake_loader.dataset.load_item(path, tgt,
                                                               margin=self.margin, crop=self.crop,
                                                               dataset_label_map=self.dlabel_map)
                images_fake = out['images']
                domain_labels_fake = out['dataset_labels']
                labels_fake = tgt

                sum_real = labels_real.shape[0]
                sum_fake = labels_fake.shape[0]
                all_images = torch.cat([images_real, images_fake], dim=0)
                all_targets = torch.cat([labels_real, labels_fake], dim=0)
                in_data, in_targets = self.to_device((all_images, all_targets))
                if domain_labels_real is not None:
                    domain_labels_real = domain_labels_real.to(self.device)
                if domain_labels_fake is not None:
                    domain_labels_fake = domain_labels_fake.to(self.device)

                # warm-up lr
                if self.warmup_step != 0 and cur_step <= self.warmup_step:
                    lr = self.config['config']['optimizer']['lr'] * float(cur_step) / self.warmup_step
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

                out_dict = self.train_unidefense_model(
                    in_data, in_targets, cur_step, grad_scalar, sum_real, sum_fake)

                cls_out = out_dict['cls_out']

                for key in out_dict.keys():
                    if "loss" in key:
                        if key not in train_loss_trackers:
                            train_loss_trackers[key] = AverageMeter()
                        train_loss_trackers[key].update(reduce_tensor(out_dict[key]).item())

                # calculate metrics
                train_acc.update(cls_out, in_targets)
                iter_acc = reduce_tensor(train_acc.mean_acc()).item()

                if self.local_rank == 0:
                    if cur_step % self.log_steps == 0:
                        log_info = {
                            "train/acc": iter_acc,
                            "train/lr": self.scheduler.get_last_lr()[0],
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

    def validate_one_split(self, val_loader, split, step):
        prob_dict = dict()
        tgt_dict = dict()
        with torch.no_grad():
            val_generator = enumerate(val_loader, 1)
            if self.local_rank == 0:
                val_generator = tqdm(val_generator, position=0, leave=True)
            for val_idx, val_data in val_generator:
                path, tgt = val_data
                out = val_loader.dataset.load_item(path, tgt, margin=self.margin, crop=self.crop)
                img_id = out['path']
                in_data = out['images'].to(self.device)

                results = self.model(in_data)
                cls_out = results["cls_out"]

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

                if self.local_rank == 0:
                    val_generator.set_description(
                        "Eval %s (%d/%d), Global Step %d" % (split, val_idx, len(val_loader), step))
        return prob_dict, tgt_dict

    def validate(self, step, timer):
        self.model.eval()

        # develop
        real_prob_dict, real_tgt_dict = self.validate_one_split(self.val_real_loader, 'real', step)
        fake_prob_dict, fake_tgt_dict = self.validate_one_split(self.val_fake_loader, 'fake', step)
        dist.barrier()

        # real and fake
        real_prob_list = [None for _ in range(dist.get_world_size())]
        fake_prob_list = [None for _ in range(dist.get_world_size())]
        real_tgt_list = [None for _ in range(dist.get_world_size())]
        fake_tgt_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(real_prob_list, real_prob_dict)
        dist.all_gather_object(fake_prob_list, fake_prob_dict)
        dist.all_gather_object(real_tgt_list, real_tgt_dict)
        dist.all_gather_object(fake_tgt_list, fake_tgt_dict)

        if self.local_rank == 0:
            real_dict = self.gather_eval_output(real_prob_list, real_tgt_list)
            real_plist, real_tlist = real_dict['frame_prob'], real_dict['frame_tgt']
            fake_dict = self.gather_eval_output(fake_prob_list, fake_tgt_list)
            fake_plist, fake_tlist = fake_dict['frame_prob'], fake_dict['frame_tgt']
            assert len(real_plist) == len(real_tlist), "prob list or tgt list for real samples corrupted"
            assert len(fake_plist) == len(fake_tlist), "prob list or tgt list for fake samples corrupted"
            metrics = cal_metrics(np.array(real_tlist + fake_tlist), np.array(real_plist + fake_plist), threshold='auto')
            print(f"Eval Step {step} [Frame], ACER {metrics['ACER']:.4f}, AUC {metrics['AUC']:.4f}, Thres {metrics['Thre']:.8f}\n"
                  f"       TP_Ratio {metrics['TP_Ratio']:.4f}, #Pos {metrics['NumP']}, "
                  f"TN_Ratio {metrics['TN_Ratio']:.4f}, #Neg {metrics['NumN']}")

        # test
        prob_dict, tgt_dict = self.validate_one_split(self.test_loader, 'test', step)
        dist.barrier()

        prob_list = [None for _ in range(dist.get_world_size())]
        tgt_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(prob_list, prob_dict)
        dist.all_gather_object(tgt_list, tgt_dict)

        if self.local_rank == 0:
            out_dict = self.gather_eval_output(prob_list, tgt_list)
            plist = out_dict['video_prob']
            tlist = out_dict['video_tgt']
            assert len(plist) == len(tlist), "prob list or tgt list corrupted"
            test_metrics_video = cal_metrics(np.array(tlist), np.array(plist), threshold=metrics['Thre'])
            print(f"Test Step {step} [Video], EER {test_metrics_video['EER']:.4f}, APCER {test_metrics_video['APCER']:.4f}, "
                  f"BPCER {test_metrics_video['BPCER']:.4f}, ACER {test_metrics_video['ACER']:.4f}, TPR5% {test_metrics_video['TPR5%']:.4f}, "
                  f"AUC {test_metrics_video['AUC']:.4f}, Thres {test_metrics_video['Thre']:.8f}\n"
                  f"       TP_Ratio {test_metrics_video['TP_Ratio']:.4f}, #Pos {test_metrics_video['NumP']}, "
                  f"TN_Ratio {test_metrics_video['TN_Ratio']:.4f}, #Neg {test_metrics_video['NumN']}")

            plist_frame = out_dict['frame_prob']
            tlist_frame = out_dict['frame_tgt']
            assert len(plist_frame) == len(tlist_frame), "prob list or tgt list corrupted"
            test_metrics_frame = cal_metrics(np.array(tlist_frame), np.array(plist_frame), threshold=metrics['Thre'])
            print(f"Test Step {step} [Frame], EER {test_metrics_frame['EER']:.4f}, APCER {test_metrics_frame['APCER']:.4f}, "
                  f"BPCER {test_metrics_frame['BPCER']:.4f}, ACER {test_metrics_frame['ACER']:.4f}, TPR5% {test_metrics_frame['TPR5%']:.4f}, "
                  f"AUC {test_metrics_frame['AUC']:.4f}, Thres {test_metrics_frame['Thre']:.8f}\n"
                  f"       TP_Ratio {test_metrics_frame['TP_Ratio']:.4f}, #Pos {test_metrics_frame['NumP']}, "
                  f"TN_Ratio {test_metrics_frame['TN_Ratio']:.4f}, #Neg {test_metrics_frame['NumN']}")

            # record the best acer and the corresponding step
            if test_metrics_frame['ACER'] < self.best_hter_frame:
                self.best_auc_frame = test_metrics_frame['AUC']
                self.best_auc_video = test_metrics_video['AUC']
                self.best_hter_frame = test_metrics_frame['ACER']
                self.best_hter_video = test_metrics_video['ACER']
                self.best_thres = test_metrics_frame['Thre']
                self.best_step = step
                self._save_ckpt(step, best=True)
            print("Best Step %d, Best AUC F %.4f, Best ACER F %.4f, Best AUC V %.4f, Best ACER V %.4f, Best Thres %.8f, "
                  "Running Time: %s, Estimated Time: %s" % (
                      self.best_step, self.best_auc_frame, self.best_hter_frame, 
                      self.best_auc_video, self.best_hter_video, self.best_thres,
                      timer.measure(), timer.measure(step / self.num_steps)
                  ))
            self._save_ckpt(step, best=False)

            val_info = {
                "val/AUC": test_metrics_frame['AUC'],
                "val/HTER": test_metrics_frame['ACER'],
                "val/TPR@5%": test_metrics_frame['TPR5%'],
                "val/best_AUC": self.best_auc_frame,
                "val/best_AUC_video": self.best_auc_video,
                "val/best_HTER": self.best_hter_frame,
                "val/best_HTER_video": self.best_hter_video
            }
            self._log_wandb(val_info, step)

    def test(self):
        self.model.eval()
        
        print(f"==> Start validation. Real: {len(self.val_real_loader)}, Fake: {len(self.val_fake_loader)}.\n"
              "\tPlease wait patiently as there will be no outputs until this is done ...")
        # develop
        real_prob_dict, real_tgt_dict = self.validate_one_split(self.val_real_loader, 'real', -1)
        fake_prob_dict, fake_tgt_dict = self.validate_one_split(self.val_fake_loader, 'fake', -1)

        real_dict = self.get_eval_output(real_prob_dict, real_tgt_dict)
        real_plist, real_tlist = real_dict['frame_prob'], real_dict['frame_tgt']
        fake_dict = self.get_eval_output(fake_prob_dict, fake_tgt_dict)
        fake_plist, fake_tlist = fake_dict['frame_prob'], fake_dict['frame_tgt']
        assert len(real_plist) == len(real_tlist), "prob list or tgt list for real samples corrupted"
        assert len(fake_plist) == len(fake_tlist), "prob list or tgt list for fake samples corrupted"
        metrics = cal_metrics(np.array(real_tlist + fake_tlist), np.array(real_plist + fake_plist), threshold='auto')
        print(f"Eval [Frame], ACER {metrics['ACER']:.4f}, AUC {metrics['AUC']:.4f}, Thres {metrics['Thre']:.8f}\n"
              f"       TP_Ratio {metrics['TP_Ratio']:.4f}, #Pos {metrics['NumP']}, "
              f"TN_Ratio {metrics['TN_Ratio']:.4f}, #Neg {metrics['NumN']}")

        print(f"==> Start testing. Test: {len(self.test_loader)}.\n"
                "\tPlease wait patiently as there will be no outputs until this is done ...")
        # test
        prob_dict, tgt_dict = self.validate_one_split(self.test_loader, 'test', -1)
        out_dict = self.get_eval_output(prob_dict, tgt_dict)
        plist = out_dict['video_prob']
        tlist = out_dict['video_tgt']
        assert len(plist) == len(tlist), "prob list or tgt list corrupted"
        test_metrics_video = cal_metrics(np.array(tlist), np.array(plist), threshold=metrics['Thre'])
        print(f"Test [Video], EER {test_metrics_video['EER']:.4f}, APCER {test_metrics_video['APCER']:.4f}, "
                f"BPCER {test_metrics_video['BPCER']:.4f}, ACER {test_metrics_video['ACER']:.4f}, TPR5% {test_metrics_video['TPR5%']:.4f}, "
                f"AUC {test_metrics_video['AUC']:.4f}, Thres {test_metrics_video['Thre']:.8f}\n"
                f"       TP_Ratio {test_metrics_video['TP_Ratio']:.4f}, #Pos {test_metrics_video['NumP']}, "
                f"TN_Ratio {test_metrics_video['TN_Ratio']:.4f}, #Neg {test_metrics_video['NumN']}")

        plist_frame = out_dict['frame_prob']
        tlist_frame = out_dict['frame_tgt']
        assert len(plist_frame) == len(tlist_frame), "prob list or tgt list corrupted"
        test_metrics_frame = cal_metrics(np.array(tlist_frame), np.array(plist_frame), threshold=metrics['Thre'])
        print(f"Test [Frame], EER {test_metrics_frame['EER']:.4f}, APCER {test_metrics_frame['APCER']:.4f}, "
                f"BPCER {test_metrics_frame['BPCER']:.4f}, ACER {test_metrics_frame['ACER']:.4f}, TPR5% {test_metrics_frame['TPR5%']:.4f}, "
                f"AUC {test_metrics_frame['AUC']:.4f}, Thres {test_metrics_frame['Thre']:.8f}\n"
                f"       TP_Ratio {test_metrics_frame['TP_Ratio']:.4f}, #Pos {test_metrics_frame['NumP']}, "
                f"TN_Ratio {test_metrics_frame['TN_Ratio']:.4f}, #Neg {test_metrics_frame['NumN']}")
        
        print(f"Summary:")
        print(f"[Video] ACER {test_metrics_video['ACER']:.4f},\tAUC {test_metrics_video['AUC']:.4f}.")
        print(f"[Frame] ACER {test_metrics_frame['ACER']:.4f},\tAUC {test_metrics_frame['AUC']:.4f}.")
