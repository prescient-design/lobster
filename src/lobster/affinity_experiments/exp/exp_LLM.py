import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from numerize import numerize
from torch import optim
from torch.utils.data import DataLoader

from lobster.affinity_experiments.models import ESM, PLM, RLM
from lobster.affinity_experiments.utils.dataset import Affinity
from lobster.affinity_experiments.utils.metric import _evaluate_binary_classifier
from lobster.affinity_experiments.utils.tools import EarlyStopping, count_parameters

from .exp_basic import ExpBasic

warnings.filterwarnings("ignore")


class ExpLLM(ExpBasic):
    """Baseline experiments for affinity data."""

    def __init__(self, args):
        super(ExpLLM, self).__init__(args)

    def _build_model(self):
        """Function that creates a model instance based on the model name.
        Returns:
            model: An instance of the model.
        """

        if self.args.task == "regression":
            self.is_classification = False
            output_size = 1
        else:
            self.is_classification = True
            self.softmax = nn.Softmax(dim=1)
            output_size = 2

        if self.args.use_moe:
            self.num_experts = self.args.num_experts

        if "esm" in self.args.model:

            model = ESM(
                self.args.model,
                use_mlp=self.args.use_mlp,
                output_size=output_size,
                use_moe=self.args.use_moe,
                num_experts=self.args.num_experts,
                gate_type=self.args.gate_type,
                expert_type=self.args.expert_type,
            )

        elif "RLM" in self.args.model:

            model = RLM(
                self.args.model,
                load_pretrained=self.args.load_pretrained,
                pretrained_path=self.args.pretrained_path,
                output_size=output_size,
                feature_token=self.args.feature_token,
                use_moe=self.args.use_moe,
                num_experts=self.args.num_experts,
                gate_type=self.args.gate_type,
                expert_type=self.args.expert_type,
            )

        elif "PLM" in self.args.model:
            model = PLM(
                self.args.model,
                load_pretrained=self.args.load_pretrained,
                pretrained_path=self.args.pretrained_path,
                output_size=output_size,
                use_moe=self.args.use_moe,
                num_experts=self.args.num_experts,
                gate_type=self.args.gate_type,
                expert_type=self.args.expert_type,
            )

        else:
            raise NotImplementedError

        if self.args.load_checkpoint:
            print("loading model....")
            model.load_state_dict(torch.load(self.args.checkpoint_path))
        self.start_epoch = self.args.start_epoch

        wandb.watch(model)
        print("Number of parameters: ", numerize.numerize(count_parameters(model)))
        return model

    def _get_data(self, flag):
        """Function that creats a dataloader basd on flag.
        Args:
            flag: Flag indicating if we should return training/validation/testing
                dataloader
        Returns:
            data_loader: Dataloader for the required dataset.
        """

        if flag == "pred" or flag == "val":
            shuffle_flag = False
            drop_last = False
        else:
            shuffle_flag = True
            drop_last = True

        data = Affinity(self.args, flag=flag)

        data_loader = DataLoader(
            data,
            batch_size=self.args.batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers
            if flag != "pred"
            else 0,  # TODO @zadorozk remove this temporary fix (num_workers > 0 caused an error: `Can't pickle local object 'Settings._validator_factory.<locals>.helper'`)
            drop_last=drop_last,
        )

        return data_loader

    def _select_optimizer(self):
        """Function that returns the optimizer based on learning rate.
        Returns:
                model_optim: model optimizer
        """
        if not self.init_step:
            print("Finetuning")
        else:
            print("Linear pruning")
            if "RLM" in self.args.model:
                for name, param in self.model.named_parameters():
                    if "base_model" in name:
                        param.requires_grad = False

            else:

                for name, param in self.model.named_parameters():
                    if "model_antibody" in name or "model_antigen" in name:
                        param.requires_grad = False

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print("Training:", name)

        model_optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr
        )

        return model_optim

    def _select_criterion(self):

        if self.is_classification:
            loss_criterion = nn.CrossEntropyLoss()
        else:
            loss_criterion = nn.MSELoss()

        return loss_criterion

    def expert_utilization_loss(self, batch_expert_utilization, k=1):
        """Function calculates the overall expert utilization loss.
        Args:
            batch_expert_utilization: the utilization of each expert in a given batch
            k: a constant
        Returns:
             expert_utilization_loss= 1/num_experts * sum_i=1^1 [num_experts(e^-kU_i-e^-k)]
        """
        expert_utilization_loss = torch.Tensor([0]).to(self.device)
        batch_expert_utilization = batch_expert_utilization.squeeze()
        for y in range(self.num_experts):
            expert_utilization_loss += torch.exp(-k * batch_expert_utilization[y]) - torch.exp(
                -torch.Tensor([k])
            ).to(self.device)
        return expert_utilization_loss / self.num_experts

    def accuracy_loss(self, pred, true, weights):

        batch_size = pred.shape[0]

        weightedPrediced = torch.bmm(pred, weights.unsqueeze(-1)).squeeze(-1)
        negSamples = true < -0.2
        posPred = weightedPrediced > -0.2
        falsePos = torch.logical_and(negSamples, posPred).bool().int() + 1

        if not self.is_classification:
            criterion = nn.MSELoss(reduction="none")
        else:
            criterion = nn.CrossEntropyLoss()

        loss = criterion(weightedPrediced, true)

        upSampledLoss = loss * falsePos
        upSampledLoss = torch.mean(upSampledLoss)

        return upSampledLoss

    def validation_step(self, data_loader, flag="valid"):
        """Prediction Function.
        Args:
            setting: Name used to be used for prediction
            load: whether to load best model
        Returns:
            mae: Mean absolute error
            mse: Mean squared error
            rmse: Root mean squared error
            mape: Mean absolute percentage error
            mspe: Mean squared percentage error
        """

        # Get model predictions
        self.model.eval()
        total_loss = []
        criterion = self._select_criterion()
        with torch.no_grad():
            for i, (batch_x1, batch_x2, batch_y) in enumerate(data_loader):

                batch_x1_input = batch_x1["input_ids"].squeeze(1).to(self.device)
                batch_x1_mask = batch_x1["attention_mask"].squeeze(1).to(self.device)

                batch_x2_input = batch_x1["input_ids"].squeeze(1).to(self.device)
                batch_x2_mask = batch_x1["attention_mask"].squeeze(1).to(self.device)

                if self.args.use_moe:

                    expert_pred, expert_weight = self.model(
                        x1=batch_x1_input,
                        x2=batch_x2_input,
                        x1_mask=batch_x1_mask,
                        x2_mask=batch_x2_mask,
                    )

                    pred = torch.matmul(expert_pred, expert_weight.unsqueeze(-1)).squeeze()
                else:
                    pred = self.model(
                        x1=batch_x1_input,
                        x2=batch_x2_input,
                        x1_mask=batch_x1_mask,
                        x2_mask=batch_x2_mask,
                    )
                if self.is_classification:
                    batch_y = torch.argmax(batch_y, dim=1).to(self.device)
                    loss = criterion(pred, batch_y)
                    pred = self.softmax(pred)
                    softmax_pred = pred.squeeze()
                    pred = torch.argmax(pred, dim=1)

                if i == 0:
                    preds = pred.detach().cpu().numpy()
                    trues = batch_y.detach().cpu().numpy()
                    softmax_preds = softmax_pred.detach().cpu().numpy()
                else:
                    preds = np.concatenate((preds, pred.detach().cpu().numpy()), axis=0)
                    softmax_preds = np.concatenate(
                        (softmax_preds, softmax_pred.detach().cpu().numpy()), axis=0
                    )
                    trues = np.concatenate((trues, batch_y.cpu().detach().numpy()), axis=0)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        if self.args.scale:
            # Transform dataset back to orignal form
            preds = self.label_scaler.inverse_transform(preds.reshape(-1, 1))
            trues = self.label_scaler.inverse_transform(trues.reshape(-1, 1))
        else:
            preds = preds.reshape(-1, 1)
            trues = trues.reshape(-1, 1)

        top_1 = softmax_preds.argmax(-1)
        pos_scores = softmax_preds[:, 1]
        out_metrics = dict()

        if not self.args.is_binary:
            # Evaluate the model performance
            nrmse, spearman_rho = metric(preds, trues)
            out_metrics.update(
                nrmse=nrmse,
                spearman_rho=spearman_rho,
            )
            out_metrics.update(records)
        else:
            binary_metrics = _evaluate_binary_classifier(softmax_preds, trues, top_1, pos_scores)
            out_metrics.update(binary_metrics)
        for k, v in out_metrics.items():
            print(flag, f"{k}: {v:.4f}")
        self.model.train()
        return out_metrics, total_loss

    def train(self, setting):
        """Training Function.
        Args:
                setting: Name used to save the model
        Returns:
                model: Trained model
        """
        # Load different datasets
        train_loader = self._get_data(flag="train")
        vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            self.args.checkpoints, patience=self.args.patience, verbose=True
        )

        # Setting optimizer and loss functions
        if self.path_for_init != "" and self.path_for_init is not None:
            self.model.module.load_state_dict(torch.load(self.path_for_init))
            print(f"LPFT Loaded {self.path_for_init}")

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model_optim, mode="min", factor=0.5, patience=3, cooldown=0, min_lr=1.0e-7
        )

        all_training_loss = []
        all_validation_loss = []

        # Training Loop
        for epoch in range(self.start_epoch, self.args.epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x1, batch_x2, batch_y) in enumerate(train_loader):

                batch_x1_input = batch_x1["input_ids"].squeeze(1).to(self.device)
                batch_x1_mask = batch_x1["attention_mask"].squeeze(1).to(self.device)

                batch_x2_input = batch_x1["input_ids"].squeeze(1).to(self.device)
                batch_x2_mask = batch_x1["attention_mask"].squeeze(1).to(self.device)
                batch_y = batch_y.to(self.device)

                batch_size = batch_x1_input.shape[0]

                iter_count += 1
                model_optim.zero_grad()
                if self.args.use_moe:
                    pred, weights = self.model(
                        x1=batch_x1_input,
                        x2=batch_x2_input,
                        x1_mask=batch_x1_mask,
                        x2_mask=batch_x2_mask,
                    )
                    # Calcuate accuracy loss
                    accuracy_loss = self.accuracy_loss(pred, batch_y, weights)
                    # Calcuate utilization loss
                    if self.args.utilization_hp != 0:
                        batch_expert_utilization = (
                            torch.sum(weights.squeeze(-1), dim=0) / batch_size
                        )
                        expert_utilization_loss = self.expert_utilization_loss(
                            batch_expert_utilization
                        )
                    else:
                        expert_utilization_loss = 0
                    loss = (
                        self.args.accuracy_hp * accuracy_loss
                        + self.args.utilization_hp * expert_utilization_loss
                    )

                else:
                    pred = self.model(
                        x1=batch_x1_input,
                        x2=batch_x2_input,
                        x1_mask=batch_x1_mask,
                        x2_mask=batch_x2_mask,
                    )
                    loss = criterion(pred, batch_y)

                train_loss.append(loss.item())

                if i % 20 == 0:
                    print(
                        "\titers: {0}/{1}, epoch: {2} | loss: {3:.4f}".format(
                            i + 1, train_steps, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    wandb.log(
                        {
                            "train_loss": loss.item(),
                            "speed": speed,
                            # 'left_time': left_time,
                            "lr": model_optim.param_groups[0]["lr"],
                        }
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.0001)
                model_optim.step()

            early_stopping._save_checkpoint(self.model, path, append="_last")
            epoch_time_cost = time.time() - epoch_time
            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_time_cost))

            train_loss = np.average(train_loss)
            all_training_loss.append(train_loss)
            vali_dic, vali_loss = self.validation_step(vali_loader, "validation")

            all_validation_loss.append(vali_loss)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            lr_scheduler.step(train_loss)

            wandb.log(
                {
                    "epoch train loss": train_loss,
                    "epoch val loss": vali_loss,
                    "epoch val aupr": vali_dic["aupr"],
                    "epoch val acc": vali_dic["accuracy"],
                    "epoch_time_cost": epoch_time_cost,
                    "epoch": epoch,
                }
            )

            # If ran out of patience stop training
            if early_stopping.early_stop:
                print("Early stopping")
                break
        return self.model

    def predict(self, setting, load=False, option=""):
        """Prediction Function.
        Args:
            setting: Name used to be used for prediction
            load: whether to load best model
        Returns:
            mae: Mean absolute error
            mse: Mean squared error
            rmse: Root mean squared error
            mape: Mean absolute percentage error
            mspe: Mean squared percentage error
        """

        # Create prediction dataset
        pred_loader = self._get_data(flag="pred")
        # Load best model saved in the checkpoint folder
        if load:
            if self.args.get("checkpoint_path") is not None:
                best_model_path = self.args.checkpoint_path

            else:
                path = os.path.join(self.args.checkpoints, setting)
                best_model_path = path + option + ".pth"

            self.model = self._build_model()

            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            if torch.cuda.device_count() > 1:
                self.device = torch.device("cuda:0")
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = nn.DataParallel(self.model)
            self.model.to(self.device)

        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x1, batch_x2) in enumerate(pred_loader):

                batch_x1_input = batch_x1["input_ids"].squeeze(1).to(self.device)
                batch_x1_mask = batch_x1["attention_mask"].squeeze(1).to(self.device)

                batch_x2_input = batch_x1["input_ids"].squeeze(1).to(self.device)
                batch_x2_mask = batch_x1["attention_mask"].squeeze(1).to(self.device)

                if self.args.use_moe:
                    expert_pred, expert_weight = self.model(
                        x1=batch_x1_input,
                        x2=batch_x2_input,
                        x1_mask=batch_x1_mask,
                        x2_mask=batch_x2_mask,
                    )
                    pred = torch.matmul(expert_pred, expert_weight.unsqueeze(-1)).squeeze()
                else:
                    pred = self.model(
                        x1=batch_x1_input,
                        x2=batch_x2_input,
                        x1_mask=batch_x1_mask,
                        x2_mask=batch_x2_mask,
                    )

                if self.is_classification:
                    pred = self.softmax(pred)
                    softmax_pred = pred.squeeze()
                    pred = torch.argmax(pred, dim=1)

                if i == 0:
                    preds = pred.detach().cpu().numpy()
                    softmax_preds = softmax_pred.detach().cpu().numpy()
                else:
                    preds = np.concatenate((preds, pred.detach().cpu().numpy()), axis=0)
                    softmax_preds = np.concatenate(
                        (softmax_preds, softmax_pred.detach().cpu().numpy()), axis=0
                    )

        if self.args.scale:
            # Transform dataset back to orignal form
            preds = self.label_scaler.inverse_transform(preds.reshape(-1, 1))

        else:
            preds = preds.reshape(-1, 1)

        # save predictions made by model
        os.makedirs(self.args.output_dir, exist_ok=True)

        if "s3" in self.args.test_data:
            if "parquet" in self.args.test_data:
                test_df = pd.read_parquet(self.args.test_data, engine="fastparquet")
            else:
                test_df = pd.read_csv(self.args.test_data)
        else:
            test_df = pd.read_csv(self.args.data_folder + self.args.test_data)

        pred_file = self.args.output_dir + setting + ".parquet"

        if self.args.oracle == "Affinity":
            test_df.loc[
                :, self.args.target_value + "_pred_class_probs"
            ] = softmax_preds.squeeze().tolist()

        test_df.to_parquet(pred_file, index=False)
        pred_file = pred_file.replace("/data/bucket", "s3://prescient-pcluster-data")
        print("Ouput file:", pred_file)

        return test_df
