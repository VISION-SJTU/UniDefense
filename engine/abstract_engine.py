import os
import wandb
import torch
import random
import numpy as np
import torch.distributed as dist
from torch.cuda.amp import autocast
from utils.misc import center_print


class AbstractEngine(object):
    path = "engine/abstract_engine.py"
    
    def __init__(self, config, stage="Train"):
        feasible_stage = ["Train", "Test"]
        if stage not in feasible_stage:
            raise ValueError(f"stage should be in {feasible_stage}, but found '{stage}'")

        self.config = config
        model_cfg = config.get("model", None)
        data_cfg = config.get("data", None)
        config_cfg = config.get("config", None)

        self.model_name = model_cfg.pop("name", None)
        self.engine_name = "Abstract"
        self.dataset_config = None

        self.gpu = None
        self.dir = None
        self.debug = None
        self.device = None
        self.resume = None
        self.local_rank = None
        self.num_classes = None

        self.best_acc = 0.
        self.best_auc = 0.
        self.best_hter = 1.0e8
        self.best_step = 1
        self.start_step = 1

        self._initiated_settings(model_cfg, data_cfg, config_cfg)
        torch.cuda.set_device(self.local_rank)

        if stage == 'Train':
            self._train_settings(model_cfg, data_cfg, config_cfg)
            self._init_wandb(model_cfg, data_cfg, config_cfg, self.config['cfg_path'])
        if stage == 'Test':
            self._test_settings(model_cfg, data_cfg, config_cfg)

    def _mprint(self, content=""):
        if self.local_rank == 0:
            print(content)

    def _initiated_settings(self, model_cfg, data_cfg, config_cfg):
        raise NotImplementedError("Not implemented in abstract class.")

    def _train_settings(self, model_cfg, data_cfg, config_cfg):
        raise NotImplementedError("Not implemented in abstract class.")

    def _test_settings(self, model_cfg, data_cfg, config_cfg):
        raise NotImplementedError("Not implemented in abstract class.")

    def _save_ckpt(self, step, best=False):
        raise NotImplementedError("Not implemented in abstract class.")

    def _load_ckpt(self, best=False, train=False):
        raise NotImplementedError("Not implemented in abstract class.")

    def to_device(self, items):
        return [obj.to(self.device) for obj in items]

    def _init_wandb(self, model_cfg, data_cfg, config_cfg, cfg_path):
        if self.local_rank == 0:
            # init wandb session
            print("Using wandb for logging.")
            wandb.init(
                dir=self.dir,
                project="UniDefense",
                group=self.engine_name,
                name=f"{self.model_name}/{self.run_id}",
                notes=config_cfg.get("notes")
            )
            wandb.config.update(
                {"model": model_cfg,
                 "config": config_cfg,
                 "data": data_cfg,
                 "dataset": self.dataset_config},
                False
            )

            # copy the script for the training model
            os.system("cp " + self.model_without_ddp.path + " " + os.path.join(wandb.run.dir, "code"))
            # copy the script for the engine
            os.system("cp " + self.path + " " + os.path.join(wandb.run.dir, "code"))
            # copy the config file
            os.system("cp " + cfg_path + " " + os.path.join(wandb.run.dir, "code"))

    @staticmethod
    def _log_wandb(log_info, step):
        wandb.log(log_info, step=step)

    @staticmethod
    def _draw_wandb(figure_name, figure, step):
        fig = {figure_name: wandb.Image(figure, caption=f"step: {step}")}
        wandb.log(fig, step=step)

    @staticmethod
    def _end_wandb():
        wandb.finish()
        center_print("Training process ends.")

    @staticmethod
    def fixed_randomness(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    def train(self):
        raise NotImplementedError("Not implemented in abstract class.")

    def validate(self, step, timer):
        raise NotImplementedError("Not implemented in abstract class.")

    def test(self):
        raise NotImplementedError("Not implemented in abstract class.")

    def train_normal_model(self, in_data, in_tgt, cur_step, grad_scalar, sum_real=None, sum_fake=None):
        with autocast(enabled=False):
            # forward pass
            out_dict = self.model(in_data, sum_real=sum_real, sum_fake=sum_fake)
            cls_out = out_dict["cls_out"]
            loss_dict = out_dict.get("loss_dict", dict())

            # asymmetrical weighted triplet loss
            if loss_dict.get("triplet") is not None:
                triplet_loss = sum([self.loss_criterion["triplet"](feat, in_tgt)
                                    for feat in loss_dict["triplet"]])
            else: 
                triplet_loss = torch.tensor(0., device=self.device)
            
            # ae loss
            if loss_dict.get("spatial") is not None:
                real_rec_loss = torch.mean(loss_dict["spatial"].narrow(0, 0, sum_real))
                fake_rec_loss = torch.mean(loss_dict["spatial"].narrow(0, sum_real, sum_fake))
            else:
                real_rec_loss = torch.tensor(0., device=self.device)
                fake_rec_loss = torch.tensor(0., device=self.device)
            
            # freq loss
            if loss_dict.get("freq") is not None:
                real_freq_loss = torch.mean(loss_dict["freq"].narrow(0, 0, sum_real))
                fake_freq_loss = torch.mean(loss_dict["freq"].narrow(0, sum_real, sum_fake))
            else:
                real_freq_loss = torch.tensor(0., device=self.device)
                fake_freq_loss = torch.tensor(0., device=self.device)

            # cross-entropy loss
            if cls_out.shape[-1] == 1:
                cls_loss = self.loss_criterion["softmax"](cls_out.squeeze(), in_tgt.float())
            else:
                cls_loss = self.loss_criterion["softmax"](cls_out, in_tgt)

            # backward
            total_loss = cls_loss + \
                self.config["config"].get("lambda_triplet", 1.) * triplet_loss + \
                self.config["config"].get("lambda_recons", 1.) * real_rec_loss + \
                self.config["config"].get("lambda_freq", 1.) * real_freq_loss

            # augment for cls loss
            if loss_dict.get("aux_cls_loss") is not None:
                total_loss += self.config["config"]["lambda_aux_cls"] * loss_dict["aux_cls_loss"]
            # augment for spatial loss, please note we **do not** split real/fake here,
            # the given 'aux_spatial' is supposed to contain only real samples
            if loss_dict.get("aux_spatial") is not None:
                aux_spatial_loss = torch.mean(loss_dict["aux_spatial"])
                total_loss += 0.1 * self.config["config"]["lambda_recons"] * aux_spatial_loss
            # augment for freq loss, please note we **do not** split real/fake here,
            # the given 'aux_freq' is supposed to contain only real samples
            if loss_dict.get("aux_freq") is not None:
                aux_freq_loss = torch.mean(loss_dict["aux_freq"])
                total_loss += 0.1 * self.config["config"]["lambda_freq"] * aux_freq_loss
            
            out_dict = {
                "total_loss": total_loss,
                "cls_out": cls_out,
                "cls_loss": cls_loss,
                "triplet_loss": triplet_loss,
                "real_rec_loss": real_rec_loss,
                "fake_rec_loss": fake_rec_loss,
                "real_freq_loss": real_freq_loss,
                "fake_freq_loss": fake_freq_loss
            }

        # backward
        grad_scalar.scale(total_loss).backward()
        grad_scalar.step(self.optimizer)
        grad_scalar.update()
        if self.warmup_step == 0 or cur_step > self.warmup_step:
            self.scheduler.step()
        dist.barrier()
        return out_dict

    def train_unidefense_model(self, in_data, in_tgt, cur_step, grad_scalar, sum_real=None, sum_fake=None):
        with autocast(enabled=False):
            # forward pass
            out_dict = self.model(in_data)
            cls_out = out_dict["cls_out"]
            loss_dict = out_dict.get("loss_dict", dict())

            # freq mask gt is optional
            if loss_dict.get('freq_mask') is not None:
                freq_mask_gt = loss_dict['freq_mask'].clone().detach()
                freq_mask_loss = torch.mean(loss_dict['freq_mask'])
            else:
                freq_mask_gt = None
                freq_mask_loss = torch.tensor(0., device=self.device)
            
            # spat mask gt is optional
            if loss_dict.get('spat_mask') is not None:
                spat_mask_gt = loss_dict['spat_mask'].clone().detach()
                spat_mask_loss = torch.mean(loss_dict['spat_mask'])
            else:
                spat_mask_gt = None
                spat_mask_loss = torch.tensor(0., device=self.device)

            fac_gt = loss_dict['factorization'].clone().detach()

            # asymmetrical weighted triplet loss
            if loss_dict.get("triplet") is not None:
                triplet_loss = sum([self.loss_criterion["triplet"](feat, in_tgt)
                                    for feat in loss_dict["triplet"]])
            else: 
                triplet_loss = torch.tensor(0., device=self.device)
            
            # ae loss
            if loss_dict.get("spatial") is not None:
                real_rec_loss = torch.mean(loss_dict["spatial"].narrow(0, 0, sum_real))
                fake_rec_loss = torch.mean(loss_dict["spatial"].narrow(0, sum_real, sum_fake))
            else:
                real_rec_loss = torch.tensor(0., device=self.device)
                fake_rec_loss = torch.tensor(0., device=self.device)
            
            # freq loss
            if loss_dict.get("freq") is not None:
                real_freq_loss = torch.mean(loss_dict["freq"].narrow(0, 0, sum_real))
                fake_freq_loss = torch.mean(loss_dict["freq"].narrow(0, sum_real, sum_fake))
            else:
                real_freq_loss = torch.tensor(0., device=self.device)
                fake_freq_loss = torch.tensor(0., device=self.device)

            # cross-entropy loss
            if cls_out.shape[-1] == 1:
                cls_loss = self.loss_criterion["softmax"](cls_out.squeeze(), in_tgt.float())
            else:
                cls_loss = self.loss_criterion["softmax"](cls_out, in_tgt)

            # backward
            total_loss = cls_loss + \
                self.config["config"].get("lambda_mask", 1.) * freq_mask_loss + \
                self.config["config"].get("lambda_mask", 1.) * spat_mask_loss + \
                self.config["config"].get("lambda_triplet", 1.) * triplet_loss + \
                self.config["config"].get("lambda_recons", 1.) * real_rec_loss + \
                self.config["config"].get("lambda_freq", 1.) * real_freq_loss

            ret_dict = {
                "total_loss": total_loss,
                "cls_out": cls_out,
                "cls_loss": cls_loss,
                "triplet_loss": triplet_loss,
                "real_rec_loss": real_rec_loss,
                "fake_rec_loss": fake_rec_loss,
                "real_freq_loss": real_freq_loss,
                "fake_freq_loss": fake_freq_loss,
            }

        # backward
        grad_scalar.scale(total_loss).backward()
        grad_scalar.step(self.optimizer)
        grad_scalar.update()
        dist.barrier()

        with autocast(enabled=False):
            # forward pass
            pert_real_list = torch.arange(sum_real)[torch.randperm(sum_real)]
            pert_fake_list = torch.arange(sum_fake)[torch.randperm(sum_fake)]
            out_dict = self.model(in_data,
                                  pert_real_list=pert_real_list, 
                                  pert_fake_list=pert_fake_list, 
                                  preserve_color=True)
            cls_out = out_dict["cls_out"]
            loss_dict = out_dict.get("loss_dict", dict())

            freq_mask_pred = loss_dict['freq_mask']
            spat_mask_pred = loss_dict['spat_mask']
            fac_pred = loss_dict['factorization']

            # asymmetrical weighted triplet loss
            if loss_dict.get("triplet") is not None:
                triplet_loss = sum([self.loss_criterion["triplet"](feat, in_tgt)
                                    for feat in loss_dict["triplet"]])
            else: 
                triplet_loss = torch.tensor(0., device=self.device)
            
            # ae loss
            if loss_dict.get("spatial") is not None:
                real_rec_loss = torch.mean(loss_dict["spatial"].narrow(0, 0, sum_real))
                fake_rec_loss = torch.mean(loss_dict["spatial"].narrow(0, sum_real, sum_fake))
            else:
                real_rec_loss = torch.tensor(0., device=self.device)
                fake_rec_loss = torch.tensor(0., device=self.device)
            
            # freq loss
            if loss_dict.get("freq") is not None:
                real_freq_loss = torch.mean(loss_dict["freq"].narrow(0, 0, sum_real))
                fake_freq_loss = torch.mean(loss_dict["freq"].narrow(0, sum_real, sum_fake))
            else:
                real_freq_loss = torch.tensor(0., device=self.device)
                fake_freq_loss = torch.tensor(0., device=self.device)

            # cross-entropy loss
            if cls_out.shape[-1] == 1:
                cls_loss = self.loss_criterion["softmax"](cls_out.squeeze(), in_tgt.float())
            else:
                cls_loss = self.loss_criterion["softmax"](cls_out, in_tgt)
            
            # activate mask alignment loss after proper init
            if cur_step > self.num_steps * 0.1:
                if freq_mask_gt is not None:
                    freq_mask_gt = freq_mask_gt.reshape(freq_mask_gt.shape[0], -1)
                    freq_mask_gt = torch.log_softmax(freq_mask_gt, dim=-1)
                    freq_mask_pred = freq_mask_pred.reshape(freq_mask_pred.shape[0], -1)
                    freq_mask_pred = torch.log_softmax(freq_mask_pred, dim=-1)
                    freq_mask_loss = self.loss_criterion["kl_div"](freq_mask_pred, freq_mask_gt)
                else:
                    freq_mask_loss = torch.zeros_like(cls_loss)
                
                if spat_mask_gt is not None:
                    spat_mask_gt = spat_mask_gt.reshape(spat_mask_gt.shape[0], -1)
                    spat_mask_gt = torch.log_softmax(spat_mask_gt, dim=-1)
                    spat_mask_pred = spat_mask_pred.reshape(spat_mask_pred.shape[0], -1)
                    spat_mask_pred = torch.log_softmax(spat_mask_pred, dim=-1)
                    spat_mask_loss = self.loss_criterion["kl_div"](spat_mask_pred, spat_mask_gt)
                else:
                    spat_mask_loss = torch.zeros_like(cls_loss)
            else:
                if freq_mask_gt is not None:
                    freq_mask_loss = torch.mean(loss_dict['freq_mask'])  
                else:  
                    freq_mask_loss = torch.zeros_like(cls_loss)
                if spat_mask_gt is not None:
                    spat_mask_loss = torch.mean(loss_dict['spat_mask'])  
                else: 
                    spat_mask_loss = torch.zeros_like(cls_loss)
            
            fac_loss = self.loss_criterion["fac"](fac_pred, fac_gt)
            ret_dict.update({"freq_mask_loss": freq_mask_loss})
            ret_dict.update({"spat_mask_loss": spat_mask_loss})
            ret_dict.update({"fac_loss": fac_loss})

            # backward
            total_loss = 0.1 * cls_loss + \
                self.config["config"].get("lambda_mask", 1.) * freq_mask_loss + \
                self.config["config"].get("lambda_mask", 1.) * spat_mask_loss + \
                self.config["config"].get("lambda_triplet", 1.) * triplet_loss + \
                self.config["config"].get("lambda_recons", 1.) * 0.1 * real_rec_loss + \
                self.config["config"].get("lambda_freq", 1.) * 0.1 * real_freq_loss + \
                self.config["config"].get("lambda_fac", 1.) * fac_loss
        
        # backward
        grad_scalar.scale(total_loss).backward()
        grad_scalar.step(self.optimizer)
        grad_scalar.update()
        if self.warmup_step == 0 or cur_step > self.warmup_step:
            self.scheduler.step()
        dist.barrier()

        return ret_dict

    @staticmethod
    def gather_eval_output(all_prob_list, all_tgt_list):
        final_prob_dict = dict()
        final_prob_dict.update(all_prob_list[0])
        for dist_id in range(dist.get_world_size()):
            if dist_id == 0:
                continue
            for i in all_prob_list[dist_id].keys():
                if i in final_prob_dict.keys():
                    final_prob_dict[i].extend(all_prob_list[dist_id][i])
                else:
                    final_prob_dict.update({i: all_prob_list[dist_id][i]})

        final_tgt_dict = dict()
        final_tgt_dict.update(all_tgt_list[0])
        for dist_id in range(dist.get_world_size()):
            if dist_id == 0:
                continue
            for i in all_tgt_list[dist_id].keys():
                if i in final_tgt_dict.keys():
                    final_tgt_dict[i].extend(all_tgt_list[dist_id][i])
                else:
                    final_tgt_dict.update({i: all_tgt_list[dist_id][i]})

        video_prob_list = list()
        video_tgt_list = list()
        frame_prob_list = list()
        frame_tgt_list = list()
        for key in final_prob_dict.keys():
            avg_single_video_prob = sum(final_prob_dict[key]) / len(final_prob_dict[key])
            avg_single_video_tgt = sum(final_tgt_dict[key]) / len(final_tgt_dict[key])
            video_prob_list.append(avg_single_video_prob)
            video_tgt_list.append(avg_single_video_tgt)
            frame_prob_list.extend(final_prob_dict[key])
            frame_tgt_list.extend(final_tgt_dict[key])
        
        out = {
            "video_prob": video_prob_list,
            "video_tgt": video_tgt_list,
            "frame_prob": frame_prob_list,
            "frame_tgt": frame_tgt_list
        }

        return out
    
    @staticmethod
    def get_eval_output(prob_dict, tgt_dict):
        video_prob_list = list()
        video_tgt_list = list()
        frame_prob_list = list()
        frame_tgt_list = list()
        for key in prob_dict.keys():
            avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
            avg_single_video_tgt = sum(tgt_dict[key]) / len(tgt_dict[key])
            video_prob_list.append(avg_single_video_prob)
            video_tgt_list.append(avg_single_video_tgt)
            frame_prob_list.extend(prob_dict[key])
            frame_tgt_list.extend(tgt_dict[key])
        
        out = {
            "video_prob": video_prob_list,
            "video_tgt": video_tgt_list,
            "frame_prob": frame_prob_list,
            "frame_tgt": frame_tgt_list
        }

        return out

    def plot_figure(self, images, items, pred, gt, cmap=None, categories=None):
        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def convert_for_plt(img):
            if img.device == "cpu":
                return img.permute([1, 2, 0]).squeeze().numpy()
            else:
                return img.permute([1, 2, 0]).detach().cpu().squeeze().numpy()

        import matplotlib.pyplot as plt
        ret = plt.figure(dpi=200)
        pred = pred.argmax(1).cpu().numpy()
        if categories is not None:
            pred = [categories[i] for i in pred]
            gt = [categories[i] for i in gt]
        num_row = len(items)
        assert len(images) == 4 * num_row, "Only display 4 samples of each item"
        if cmap is None:
            cmap = [None] * num_row

        plt.title("pred: %s\ngt: %s\nitem: %s" % (pred, gt, items))
        for i in range(num_row):
            for j in range(4):
                ax = ret.add_subplot(num_row, 4, i * 4 + j + 1)
                ax.axis("off")
                norm_ip(images[i * 4 + j], float(images[i * 4 + j].min()), float(images[i * 4 + j].max()))
                if cmap[i] is None:
                    ax.imshow(convert_for_plt(images[i * 4 + j]))
                else:
                    ax.imshow(convert_for_plt(images[i * 4 + j]), cmap=cmap[i])

        plt.axis("off")
        plt.close()
        return ret
