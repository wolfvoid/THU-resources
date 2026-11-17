import copy

import numpy as np


def moving_avg(x, kernel_size, stride):
    """
    Moving average to highlight the trend of time series

    :param x: [bs, seq_len, num_dim]
    :param kernel_size: int
    :param stride: int
    """
    bs, seq_len, num_dim = x.shape

    # padding on the both ends of time series
    front = np.tile(x[:, 0:1, :], (1, (kernel_size - 1) // 2, 1))
    end = np.tile(x[:, -1:, :], (1, (kernel_size - 1) // 2, 1))
    x = np.concatenate([front, x, end], axis=1)
    x = np.transpose(x, (0, 2, 1)).reshape(-1, x.shape[1])  # [B, T, N] => [BxN, T]
    # 1d avg pooling
    x = np.stack([x[:, i:i + kernel_size].mean(-1) for i in range(0, seq_len, stride)], axis=-1)
    x = x.reshape(bs, -1, seq_len).transpose(0, 2, 1)
    return x


def series_decomp(x, kernel_size=25):
    """
    Series decomposition

    :param x: [bs, seq_len, num_dim]
    :param kernel_size: int
    """
    moving_mean = moving_avg(x, kernel_size, stride=1)
    res = x - moving_mean
    return res, moving_mean


class Model:

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.hidden = configs.hidden_size
        self.init_weights()

        self.check_grad()

    def init_weights(self):
        """Initialize weights."""

        #######################################
        # TODO: weight initialization
        # NOTE: proper initialization can be very important

        # w1 ∈ X^(T*H),  w2 ∈ X^(H*S),
        # b1 ∈ X^H， b2 ∈ X^S，

        # He initialization (normal distribution)
        std_s1 = np.sqrt(2.0 / self.seq_len)
        std_t1 = np.sqrt(2.0 / self.seq_len)
        std_s2 = np.sqrt(1.0 / self.hidden)
        std_t2 = np.sqrt(1.0 / self.hidden)

        # Initialize weights with random normal distribution
        self.w_s1 = np.random.normal(0, std_s1, size=(self.seq_len, self.hidden))
        self.w_t1 = np.random.normal(0, std_t1, size=(self.seq_len, self.hidden))
        self.w_s2 = np.random.normal(0, std_s2, size=(self.hidden, self.pred_len))
        self.w_t2 = np.random.normal(0, std_t2, size=(self.hidden, self.pred_len))

        # 使用固定的标准差，比如0.01或0.02（弃用）
        # std = 0.01
        # self.w_s1 = np.random.normal(0, std, size=(self.seq_len, self.hidden))
        # self.w_t1 = np.random.normal(0, std, size=(self.seq_len, self.hidden))
        # self.w_s2 = np.random.normal(0, std, size=(self.hidden, self.pred_len))
        # self.w_t2 = np.random.normal(0, std, size=(self.hidden, self.pred_len))

        # Initialize biases to 0
        self.b_s1 = np.zeros(self.hidden)
        self.b_t1 = np.zeros(self.hidden)
        self.b_s2 = np.zeros(self.pred_len)
        self.b_t2 = np.zeros(self.pred_len)
        #######################################

        # Reset gradients
        self.w_s1_grad, self.b_s1_grad = None, None
        self.w_s2_grad, self.b_s2_grad = None, None
        self.w_t1_grad, self.b_t1_grad = None, None
        self.w_t2_grad, self.b_t2_grad = None, None

    def forward_backward(self, x, y, forward_only=False):
        """Forward Propagation and Backward Propagation.

        Given a mini-batch of input sequence x, associated ground-truth y, compute the
        prediction, MSE loss and gradients corresponding to these parameters.

        :param x: input sequence np.array [bs, seq_len, num_dim]
        :param y: ground-truth np.array [bs, pred_len, num_dim]
        :param forward_only: if True, backward is not needed
        :return: pred: prediction np.array [bs, pred_len, num_dim]
        """
        batch_size = x.shape[0]
        num_dim = x.shape[2]

        # Series decomposition
        seasonal_init, trend_init = series_decomp(x)

        # Reshape: [B, T, N] => [BxN, T], all dimensions share the same network
        seasonal_init = np.transpose(seasonal_init, (0, 2, 1)).reshape(-1, self.seq_len)
        trend_init = np.transpose(trend_init, (0, 2, 1)).reshape(-1, self.seq_len)
        # print("seasonal_init shape:", seasonal_init.shape)
        # print("trend_init shape:", trend_init.shape)

        #######################################
        # TODO: forward pass
        s_1 = seasonal_init @ self.w_s1 + self.b_s1  # First layer
        s_h = np.maximum(0, s_1)  # ReLU activation
        s_2 = s_h @ self.w_s2 + self.b_s2  # Second layer

        t_1 = trend_init @ self.w_t1 + self.b_t1  # First layer
        t_h = np.maximum(0, t_1)  # ReLU activation
        t_2 = t_h @ self.w_t2 + self.b_t2  # Second layer
        #######################################

        # Compose seasonal and trend components
        x = s_2 + t_2
        # Reshape: [BxN, S] => [B, S, N]
        pred = x.reshape(batch_size, -1, self.pred_len).transpose(0, 2, 1)

        #######################################
        # TODO: calculate MSE loss
        # print ("pred shape:", pred.shape)
        # print ("y shape:", y.shape)
        loss = np.mean((pred - y) ** 2)
        #######################################

        if not forward_only:
            #######################################
            # TODO: backward pass
            # Calculate gradients
            # Gradient of loss w.r.t. pred
            # 1. 计算损失函数的梯度
            d_Y = 2 * (pred - y) / (batch_size * self.pred_len * num_dim)  # [B, S, N]
            d_Y = d_Y.transpose(0, 2, 1).reshape(-1, self.pred_len)  # [BxN, S] 即dL/dY
            # d_Yt= d_Ys = d_Y ∈ [BxN, S]

            # 2. 季节分支的反向传播
            # 第二层梯度 (s_2 = s_h @ w_s2 + b_s2)
            grad_h1_s = d_Y @ self.w_s2.T  # [BxN, H]
            self.w_s2_grad = s_h.T @ d_Y  # [H, S]
            self.b_s2_grad = np.sum(d_Y, axis=0)  # [S]

            # 第一层梯度 (h1 = ReLU(z1), z1 = seasonal_init @ w_s1 + b_s1)
            grad_z1_s = grad_h1_s * (s_1 > 0)  # ReLU梯度 [BxN, H]
            self.w_s1_grad = seasonal_init.T @ grad_z1_s  # [T, H]
            self.b_s1_grad = np.sum(grad_z1_s, axis=0)  # [H]

            # 3. 趋势分支的反向传播
            # 第二层梯度 (trend_output = h1 @ w_t2 + b_t2)
            grad_h1_t = d_Y @ self.w_t2.T  # [BxN, H]
            self.w_t2_grad = t_h.T @ d_Y  # [H, S]
            self.b_t2_grad = np.sum(d_Y, axis=0)  # [S]

            # 第一层梯度(h1 = ReLU(z1), z1 = trend_init @ w_t1 + b_t1)
            grad_z1_t = grad_h1_t * (t_1 > 0)  # ReLU梯度 [BxN, H]
            self.w_t1_grad = trend_init.T @ grad_z1_t  # [T, H]
            self.b_t1_grad = np.sum(grad_z1_t, axis=0)  # [H]
            #######################################
        return pred, loss

    def update_weights(self, lr, weight_decay):
        """Gradient Descent

        Update weights using calculated gradients.

        :param lr: learning rate
        :param weight_decay: weight decay
        """

        #######################################
        # TODO: gradient descent
        # Season branch
        self.w_s1 -= lr * (self.w_s1_grad + weight_decay * self.w_s1)
        self.b_s1 -= lr * self.b_s1_grad
        self.w_s2 -= lr * (self.w_s2_grad + weight_decay * self.w_s2)
        self.b_s2 -= lr * self.b_s2_grad
        # Trend branch
        self.w_t1 -= lr * (self.w_t1_grad + weight_decay * self.w_t1)
        self.b_t1 -= lr * self.b_t1_grad
        self.w_t2 -= lr * (self.w_t2_grad + weight_decay * self.w_t2)
        self.b_t2 -= lr * self.b_t2_grad
        #######################################

    def state_dict(self):
        return copy.deepcopy({
            'w_s1': self.w_s1,
            'b_s1': self.b_s1,
            'w_s2': self.w_s2,
            'b_s2': self.b_s2,
            'w_t1': self.w_t1,
            'b_t1': self.b_t1,
            'w_t2': self.w_t2,
            'b_t2': self.b_t2,
        })

    def load_state_dict(self, state_dict):
        self.w_s1 = state_dict['w_s1']
        self.b_s1 = state_dict['b_s1']
        self.w_s2 = state_dict['w_s2']
        self.b_s2 = state_dict['b_s2']
        self.w_t1 = state_dict['w_t1']
        self.b_t1 = state_dict['b_t1']
        self.w_t2 = state_dict['w_t2']
        self.b_t2 = state_dict['b_t2']

    def check_grad(self):
        """Check backward propagation implementation. This is naively implemented with finite difference method.
        You do **not** need to modify this function.
        """
        # store original values
        state_dict = self.state_dict()
        original_seq_len = self.seq_len
        original_pred_len = self.pred_len
        original_hidden = self.hidden
        self.seq_len = 10
        self.hidden = 25
        self.pred_len = 15

        def relative_error(z1, z2):
            return np.mean((z1 - z2) ** 2 / (z1 ** 2 + z2 ** 2))

        print('Gradient check of backward propagation:')

        # generate random test data
        x = np.random.rand(5, self.seq_len, 3)
        y = np.random.rand(5, self.pred_len, 3)

        # generate random parameters
        self.w_s1 = np.random.rand(self.seq_len, self.hidden)
        self.b_s1 = np.random.rand(self.hidden)
        self.w_s2 = np.random.rand(self.hidden, self.pred_len)
        self.b_s2 = np.random.rand(self.pred_len)
        self.w_t1 = np.random.rand(self.seq_len, self.hidden)
        self.b_t1 = np.random.rand(self.hidden)
        self.w_t2 = np.random.rand(self.hidden, self.pred_len)
        self.b_t2 = np.random.rand(self.pred_len)

        # calculate grad by backward propagation
        _, loss = self.forward_backward(x, y)

        # calculate grad by finite difference
        epsilon = 1e-5

        weights_dict = {
            'w_s1': (self.w_s1, self.w_s1_grad),
            'b_s1': (self.b_s1, self.b_s1_grad),
            'w_s2': (self.w_s2, self.w_s2_grad),
            'b_s2': (self.b_s2, self.b_s2_grad),
            'w_t1': (self.w_t1, self.w_t1_grad),
            'b_t1': (self.b_t1, self.b_t1_grad),
            'w_t2': (self.w_t2, self.w_t2_grad),
            'b_t2': (self.b_t2, self.b_t2_grad),
        }

        for name, (weight, gradient) in weights_dict.items():
            numeric = np.zeros_like(weight)
            if len(weight.shape) == 1:
                numeric = np.zeros_like(weight)
                for i in range(weight.shape[0]):
                    weight[i] += epsilon
                    _, loss_prime = self.forward_backward(x, y, forward_only=True)
                    weight[i] -= epsilon
                    numeric[i] = (loss_prime - loss) / epsilon
                re = relative_error(numeric, gradient)
                print(f'Relative error of d{name}', re)
                assert re < 1e-5, 'Gradient check failed. If you implement back propagation correctly, all these relative errors should be less than 1e-5.'
            else:
                numeric = np.zeros_like(weight)
                for i in range(weight.shape[0]):
                    for j in range(weight.shape[1]):
                        weight[i, j] += epsilon
                        _, loss_prime = self.forward_backward(x, y, forward_only=True)
                        weight[i, j] -= epsilon
                        numeric[i, j] = (loss_prime - loss) / epsilon
                re = relative_error(numeric, gradient)
                print(f'Relative error of d{name}', re)
                assert re < 1e-5, 'Gradient check failed. If you implement back propagation correctly, all these relative errors should be less than 1e-5.'

        print('Gradient check passed!')

        # restore original values
        self.load_state_dict(state_dict)
        self.seq_len = original_seq_len
        self.pred_len = original_pred_len
        self.hidden = original_hidden


if __name__ == '__main__':
    class Configs:
        seq_len = 4 # 96
        pred_len = 8 # 192
        hidden_size = 512

    configs = Configs()
    model = Model(configs)
    x=np.random.rand(5, configs.seq_len, 3)
    y=np.random.rand(5, configs.pred_len, 3)

    model.forward_backward(x, y, forward_only=True)
