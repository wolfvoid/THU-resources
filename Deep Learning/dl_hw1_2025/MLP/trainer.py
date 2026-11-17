import os
import pickle
import time
import warnings

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import model
from data_loader import data_provider
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from datetime import datetime

warnings.filterwarnings('ignore')

def create_experiment_dir(base_dir, experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cpu')
        self.model = model.Model(self.args)
        exp_dir = create_experiment_dir('runs', args.model)
        self.writer = SummaryWriter(log_dir=exp_dir)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def vali(self, vali_data, vali_loader):
        total_loss = []
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            # for this homework, we use torch for data loading only
            batch_x = batch_x.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            outputs, loss = self.model.forward_backward(batch_x, batch_y, forward_only=True)

            total_loss.append(loss)
        total_loss = np.average(total_loss)
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        print(f"Train samples: {len(train_data)}")
        print(f"Validation samples: {len(vali_data)}")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(verbose=True)
        lr, wd = self.args.learning_rate, self.args.weight_decay

        # 添加全局步数计数器
        global_step = 0

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                global_step += 1

                # for this homework, we use torch for data loading only
                batch_x = batch_x.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                outputs, loss = self.model.forward_backward(batch_x, batch_y)
                self.model.update_weights(lr, wd)

                train_loss.append(loss)

                # 每个batch都记录训练损失到TensorBoard
                self.writer.add_scalar('Loss/train_batch', loss.item(), global_step)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            # 记录每个epoch的训练和验证损失
            self.writer.add_scalar('Loss/train_epoch', train_loss, epoch + 1)
            self.writer.add_scalar('Loss/val_epoch', vali_loss, epoch + 1)
            self.writer.add_scalar('Learning_rate', lr, epoch + 1)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            lr = adjust_learning_rate(epoch + 1, self.args)

        # 关闭TensorBoard writer
        self.writer.close()

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(pickle.load(open(best_model_path, 'br')))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(pickle.load(
                open(os.path.join('./checkpoints/' + setting, 'checkpoint.pth', 'br'))))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for i, (batch_x, batch_y) in enumerate(test_loader):
            # for this homework, we use torch for data loading only
            batch_x = batch_x.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            outputs, loss = self.model.forward_backward(batch_x, batch_y, forward_only=True)

            pred = outputs
            true = batch_y

            preds.append(pred)
            trues.append(true)
            if i % 20 == 0:
                input = batch_x
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # 记录测试指标到TensorBoard
        self.writer.add_scalar('Test/mse', mse, 0)
        self.writer.add_scalar('Test/mae', mae, 0)
        self.writer.add_scalar('Test/rmse', rmse, 0)

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
