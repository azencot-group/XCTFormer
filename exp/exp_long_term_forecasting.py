import json
import os
import sys
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


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                         weight_decay=self.args.weight_decay)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_mse_loss = []
        total_mae_loss = []
        batch_sizes = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                mae, mse, rmse, mape, mspe = metric(pred, true)

                total_mse_loss.append(mse)
                total_mae_loss.append(mae)
                batch_sizes.append(pred.shape[0])

        batch_sizes = np.array(batch_sizes)
        batch_sizes = batch_sizes / np.sum(batch_sizes)
        total_mse_loss = np.sum(np.array(total_mse_loss) * batch_sizes)
        total_mae_loss = np.sum(np.array(total_mae_loss) * batch_sizes)
        self.model.train()
        return total_mse_loss, total_mae_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

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

        best_validation_loss = sys.maxsize

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            if (epoch + 1) % self.args.handle_end_epoch_rate != 0:
                continue

            train_loss = np.average(train_loss)

            vali_loss, vali_mae_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mae_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} Test MAE: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss, test_mae_loss))

            outputs = self.model(batch_x)
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            loss = criterion(outputs, batch_y)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

            if vali_loss < best_validation_loss:
                best_validation_loss = vali_loss
                torch.save(self.model.state_dict(),
                           os.path.join(self.checkpoint_path, 'checkpoint_best.pth'))

        self.test(setting)
        return best_validation_loss

    def test(self, setting, test=1):
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
        batch_sizes = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                mae, mse, rmse, mape, mspe = metric(outputs, batch_y)

                total_mae_losses.append(mae)
                total_mse_losses.append(mse)
                total_rmse_losses.append(rmse)
                total_mape_losses.append(mape)
                total_mspe_losses.append(mspe)
                batch_sizes.append(outputs.shape[0])

        batch_sizes = np.array(batch_sizes)
        weights = batch_sizes / np.sum(batch_sizes)

        final_mae = np.sum(np.array(total_mae_losses) * weights)
        final_mse = np.sum(np.array(total_mse_losses) * weights)
        final_rmse = np.sum(np.array(total_rmse_losses) * weights)
        final_mape = np.sum(np.array(total_mape_losses) * weights)
        final_mspe = np.sum(np.array(total_mspe_losses) * weights)

        print(f'test metrics - mse: {final_mse:.6f}, mae: {final_mae:.6f}, '
              f'rmse: {final_rmse:.6f}, mape: {final_mape:.6f}, mspe: {final_mspe:.6f}')

        results_dir = os.path.join('test_results', self.args.task_name)
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f'{self.args.model_id}.json')
        result = {"setting": setting, "mse": float(final_mse), "mae": float(final_mae)}
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=4)

        return
