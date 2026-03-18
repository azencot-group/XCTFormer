import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

warnings.filterwarnings('ignore')


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0
                mask[mask > self.args.mask_rate] = 1
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                mask = mask.detach().cpu()

                loss = criterion(pred[mask == 0], true[mask == 0])
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = self.checkpoint_path

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        best_vali_loss = 100
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0
                mask[mask > self.args.mask_rate] = 1
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        self.test(setting, 1)
        return best_vali_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(self.checkpoint_path, 'checkpoint_best.pth')))

        total_mae_losses = []
        total_mse_losses = []
        total_rmse_losses = []
        total_mape_losses = []
        total_mspe_losses = []
        masked_sample_counts = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0
                mask[mask > self.args.mask_rate] = 1
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                outputs_cpu = outputs.detach().cpu().numpy()
                true_cpu = batch_x.detach().cpu().numpy()
                mask_cpu = mask.detach().cpu().numpy()

                masked_indices = mask_cpu == 0
                batch_masked_count = np.sum(masked_indices)

                if batch_masked_count > 0:
                    masked_pred = outputs_cpu[masked_indices]
                    masked_true = true_cpu[masked_indices]
                    batch_mae, batch_mse, batch_rmse, batch_mape, batch_mspe = metric(masked_pred, masked_true)
                    total_mae_losses.append(batch_mae)
                    total_mse_losses.append(batch_mse)
                    total_rmse_losses.append(batch_rmse)
                    total_mape_losses.append(batch_mape)
                    total_mspe_losses.append(batch_mspe)
                    masked_sample_counts.append(batch_masked_count)

        masked_sample_counts = np.array(masked_sample_counts)
        total_masked_samples = np.sum(masked_sample_counts)

        if total_masked_samples > 0:
            weights = masked_sample_counts / total_masked_samples
            final_mae = np.sum(np.array(total_mae_losses) * weights)
            final_mse = np.sum(np.array(total_mse_losses) * weights)
            final_rmse = np.sum(np.array(total_rmse_losses) * weights)
            final_mape = np.sum(np.array(total_mape_losses) * weights)
            final_mspe = np.sum(np.array(total_mspe_losses) * weights)
        else:
            final_mae = final_mse = final_rmse = final_mape = final_mspe = 0.0

        print(f'test metrics - mse: {final_mse:.6f}, mae: {final_mae:.6f}')

        results_dir = os.path.join('test_results', self.args.task_name)
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f'{self.args.model_id}.json')
        result = {"setting": setting, "mse": float(final_mse), "mae": float(final_mae)}
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=4)

        return
