import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import NNConv
from dgl.nn.pytorch.glob import MaxPooling
import dgl.function as fn
import networkx as nx
import matplotlib.pyplot as plt
from dgl.nn.pytorch import GraphConv
import dgl.nn as dglnn

# Basic models
def _conv1d(in_channels, out_channels, kernel_size=3, padding=0, bias=False):
    """
    Helper function to create a 1D convolutional layer with batchnorm and LeakyReLU activation

    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding size on each side. Defaults to 0.
        bias (bool, optional): Whether bias is used. Defaults to False.

    Returns:
        nn.Sequential: Sequential contained the Conv1d, BatchNorm1d and LeakyReLU layers
    """
    return nn.Sequential(
        nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias
        ),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(),
    )

def _conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False):
    """
    Helper function to create a 2D convolutional layer with batchnorm and LeakyReLU activation

    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding size on each side. Defaults to 0.
        bias (bool, optional): Whether bias is used. Defaults to False.

    Returns:
        nn.Sequential: Sequential contained the Conv2d, BatchNorm2d and LeakyReLU layers
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )

def _fc(in_features, out_features, bias=False):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(),
    )

class _MLP(nn.Module):
    """"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        MLP with linear output
        Args:
            num_layers (int): The number of linear layers in the MLP
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden feature dimensions for all hidden layers
            output_dim (int): Output feature dimension

        Raises:
            ValueError: If the given number of layers is <1
        """
        super(_MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)

# geometry encoders
class UVNetCurveEncoder(nn.Module):
    def __init__(self, in_channels=6, output_dims=64):
        """
        This is the 1D convolutional network that extracts features from the B-rep edge
        geometry described as 1D UV-grids

        Args:
            in_channels (int, optional): Number of channels in the edge UV-grids. By default
                                         we expect 3 channels for point coordinates and 3 for
                                         curve tangents. Defaults to 6.
            output_dims (int, optional): Output curve embedding dimension. Defaults to 64.
        """
        super(UVNetCurveEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv1d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = _conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv3 = _conv1d(128, 256, kernel_size=3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = _fc(256, output_dims, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

class UVNetSurfaceEncoder(nn.Module):
    def __init__(
        self,
        in_channels=7,
        output_dims=64,
    ):
        """
        This is the 2D convolutional network that extracts features from the B-rep face
        geometry described as 2D UV-grids.

        Args:
            in_channels (int, optional): Number of channels in the edge UV-grids. By default
                                         we expect 3 channels for point coordinates and 3 for
                                         surface normals and 1 for the trimming mask. Defaults
                                         to 7.
            output_dims (int, optional): Output surface embedding dimension. Defaults to 64.
        """
        super(UVNetSurfaceEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.conv2 = _conv2d(64, 128, 3, padding=1, bias=False)
        self.conv3 = _conv2d(128, 256, 3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _fc(256, output_dims, bias=False)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

class _EdgeConv(nn.Module):
    def __init__(
        self,
        edge_feats,
        out_feats,
        node_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
    ):
        """
        This module update the edge features using the node features at the endpoints.

        Args:
            edge_feats (int): Input edge feature dimension
            out_feats (int): Output feature deimension
            node_feats (int): Input node feature dimension
            num_mlp_layers (int, optional): Number of layers used in the MLP. Defaults to 2.
            hidden_mlp_dim (int, optional): Hidden feature dimension in the MLP. Defaults to 64.
        """
        super(_EdgeConv, self).__init__()
        self.proj = _MLP(1, node_feats, hidden_mlp_dim, edge_feats)
        self.mlp = _MLP(num_mlp_layers, edge_feats, hidden_mlp_dim, out_feats)
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):
        src, dst = graph.edges()
        proj1, proj2 = self.proj(nfeat[src]), self.proj(nfeat[dst])
        agg = proj1 + proj2
        h = self.mlp((1 + self.eps) * efeat + agg)
        h = F.leaky_relu(self.batchnorm(h))
        return h

# Wrappers for different GNN layers
class GCNConvWrapper(nn.Module):
    def __init__(self, in_feats, out_feats, bias=False):
        super(GCNConvWrapper, self).__init__()
        self.gconv = GraphConv(
            in_feats=in_feats,
            out_feats=out_feats,
            norm='both',
            weight=True,
            bias=bias
        )

    def forward(self, g, node_feat, edge_feat=None):
        """
        Args:
            g (dgl.DGLGraph): 输入图，定义节点与边的连接关系
                - g.num_nodes() 返回图中节点总数
                - g.num_edges() 返回图中边的总数
                - g.edges() 返回两个一维张量 (src, dst)
                  其中 src[i] 表示第 i 条边的源节点索引
                  dst[i] 表示第 i 条边的目标节点索引
                - g.in_degrees(i) 返回节点 i 的入度
                - g.out_degrees(i) 返回节点 i 的出度
            node_feat (Tensor): 节点特征矩阵，形状为 (N, in_feats)
            edge_feat (Tensor, optional): 边特征（本层未使用，保留接口兼容性）

        Returns:
            Tensor: 更新后的节点特征矩阵，形状为 (N, out_feats)
        """
        return self.gconv(g, node_feat)

class GINWrapper(nn.Module):
    # TODO 2.3 Task B
    def __init__(self, in_feats, out_feats, bias=False):
        super(GINWrapper, self).__init__()
        # GIN 需要一个 MLP 来转换聚合后的特征
        # 这里使用一个简单的线性层来匹配维度，也可以使用两层 MLP (Linear->ReLU->Linear)
        apply_func = nn.Linear(in_feats, out_feats, bias=bias)
        # apply_func = nn.Sequential(
        #     nn.Linear(in_feats, out_feats, bias=bias),
        #     nn.ReLU(),
        #     nn.Linear(out_feats, out_feats, bias=bias)
        # )
        # aggregator_type 可以选 'sum', 'max', or 'mean'
        self.gin = dglnn.GINConv(apply_func, aggregator_type='sum')

    def forward(self, g, node_feat, edge_feat=None):
        return self.gin(g, node_feat)

class GATConvWrapper(nn.Module):
    # TODO 2.3 Task B
    def __init__(self, in_feats, out_feats, bias=False):
        super(GATConvWrapper, self).__init__()
        # num_heads=1 保证输出维度和 out_feats 一致
        # 如果使用多头注意力，输出维度会变成 out_feats * num_heads，需要额外处理
        self.gat = dglnn.GATConv(in_feats, out_feats, num_heads=1, bias=bias)

    def forward(self, g, node_feat, edge_feat=None):
        # GAT 输出形状为 (N, num_heads, out_feats)
        out = self.gat(g, node_feat)
        # 压缩掉 num_heads 维度，变成 (N, out_feats)
        return out.mean(dim=1)

class SAGEConvWrapper(nn.Module):
    # TODO 2.3 Task B
    def __init__(self, in_feats, out_feats, bias=False):
        super(SAGEConvWrapper, self).__init__()
        # aggregator_type 可以选 'mean', 'gcn', 'pool', 'lstm'
        self.sage = dglnn.SAGEConv(
            in_feats, out_feats, aggregator_type='pool', bias=bias)

    def forward(self, g, node_feat, edge_feat=None):
        return self.sage(g, node_feat)

class CustomConv(nn.Module):
    # TODO 2.3 Task C
    def __init__(self, in_feats, out_feats, edge_feats, num_mlp_layers=2, hidden_mlp_dim=64):
        """
        Custom Graph Convolution Layer for Task C.

        Args:
            in_feats (int): Dimension of input node features (d_in).
            out_feats (int): Dimension of output node features (d_out).
            edge_feats (int): Dimension of edge features.
            num_mlp_layers (int): Number of layers in the MLP.
            hidden_mlp_dim (int): Hidden dimension for the MLP.
        """
        super(CustomConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats

        # 1. 边的权重生成网络 (Edge Weight Generator)
        # 对应公式: W_ij = reshape(MLP(e_ij))
        # MLP 输入: edge_feats
        # MLP 输出: d_out * d_in (为了后续 reshape 成权重矩阵)
        self.edge_mlp = _MLP(num_mlp_layers, edge_feats,
                             hidden_mlp_dim, out_feats * in_feats)

        # 2. 自身的残差变换 (Residual Transformation)
        # 对应公式: W_r * h_i + b
        # nn.Linear 默认包含 bias (b)，所以这里同时实现了 W_r 和 b
        self.res_fc = nn.Linear(in_feats, out_feats, bias=True)

    def forward(self, g, h, e):
        """
        Args:
            g (DGLGraph): The graph.
            h (Tensor): Node features. Shape: (N, in_feats)
            e (Tensor): Edge features. Shape: (M, edge_feats)
        """
        with g.local_scope():  # 保证不污染原图数据
            # --- 步骤 1: 计算动态边权重 W_ij ---
            # MLP 输出形状: (num_edges, out_feats * in_feats)
            w_flat = self.edge_mlp(e)

            # Reshape 为矩阵形状: (num_edges, out_feats, in_feats)
            g.edata['w'] = w_flat.view(-1, self.out_feats, self.in_feats)

            # 将节点特征赋予图，准备消息传递
            g.ndata['h'] = h

            # --- 步骤 2 & 3: 消息计算与聚合 (Message Passing & Aggregation) ---
            # message_func (msg_fn): 计算 m_ij = W_ij * h_j
            # reduce_func (reduce_fn): 计算 sum_{j in N(i)} m_ij
            g.update_all(self.msg_fn, self.reduce_fn)

            # 获取聚合后的结果 (Sum Aggregation)
            # 形状: (num_nodes, out_feats)
            h_agg_sum = g.ndata['h_agg']

            # --- 步骤 4: 归一化 (Normalization) ---
            # 对应公式: h_bar_i = (1 / deg(i)) * sum(...)
            # 获取入度，并防止除以 0
            deg = g.in_degrees().float().clamp(min=1).to(h.device).unsqueeze(1)
            h_bar = h_agg_sum / deg

            # --- 步骤 5: 残差连接与偏置 (Residual & Bias) ---
            # 对应公式: h'_i = h_bar_i + W_r * h_i + b
            h_res = self.res_fc(h)  # W_r * h_i + b
            h_prime = h_bar + h_res

            # --- 步骤 6: 激活函数 (Activation) ---
            # 对应公式: h_out = ReLU(h'_i)
            return F.relu(h_prime)

    def msg_fn(self, edges):
        """
        消息函数: 计算 m_ij = W_ij * h_j

        维度变化:
        W_ij (edges.data['w']): (Batch, out_feats, in_feats)
        h_j  (edges.src['h']): (Batch, in_feats) -> unsqueeze -> (Batch, in_feats, 1)
        Batch Matmul: (B, out, in) x (B, in, 1) -> (B, out, 1)
        Result: squeeze -> (Batch, out_feats)
        """
        w = edges.data['w']
        h = edges.src['h'].unsqueeze(2)
        # 批量矩阵乘法
        m = torch.bmm(w, h).squeeze(2)
        return {'m': m}

    def reduce_fn(self, nodes):
        """
        聚合函数: 显式求和

        维度变化:
        nodes.mailbox['m']: (Batch, Degree, out_feats)
        Result: sum over Degree -> (Batch, out_feats)
        """
        return {'h_agg': torch.sum(nodes.mailbox['m'], dim=1)}


class _NodeConv(nn.Module):
    def __init__(
        self,
        node_feats,
        out_feats,
        edge_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
        conv_type='GCN',    # 'GCN', 'GIN', 'GAT', 'SAGE', 'Custom'
    ):
        """
        This module update the node features using the neighboring node and edge features.

        Args:
            node_feats (int): Input edge feature dimension
            out_feats (int): Output feature deimension
            node_feats (int): Input node feature dimension
            num_mlp_layers (int, optional): Number of layers used in the MLP. Defaults to 2.
            hidden_mlp_dim (int, optional): Hidden feature dimension in the MLP. Defaults to 64.
        """
        super(_NodeConv, self).__init__()
        # You need to replace the gconv here with different modules.
        # TODO 2.3 Task B
        if conv_type == "GCN":
            self.gconv = GCNConvWrapper(in_feats=node_feats, out_feats=out_feats, bias=True)
        elif conv_type == "GIN":
            self.gconv = GINWrapper(in_feats=node_feats, out_feats=out_feats, bias=True)
        elif conv_type == "GAT":
            self.gconv = GATConvWrapper(in_feats=node_feats, out_feats=out_feats, bias=True)
        elif conv_type == "SAGE":
            self.gconv = SAGEConvWrapper(in_feats=node_feats, out_feats=out_feats, bias=True)
        elif conv_type == "Custom":
            self.gconv = CustomConv(
                in_feats=node_feats,
                out_feats=out_feats,
                edge_feats=edge_feats,
                num_mlp_layers=num_mlp_layers,
                hidden_mlp_dim=hidden_mlp_dim)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.mlp = _MLP(num_mlp_layers, node_feats, hidden_mlp_dim, out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):
        h = (1 + self.eps) * nfeat
        h = self.gconv(graph, h, efeat)
        h = self.mlp(h)
        h = F.leaky_relu(self.batchnorm(h))
        return h

class UVNetGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        input_edge_dim,
        output_dim,
        hidden_dim=64,
        learn_eps=True,
        num_layers=3,
        num_mlp_layers=2,
        conv_type='GCN',    # 'GCN', 'GIN', 'GAT', 'SAGE', 'Custom'
    ):
        """
        This is the graph neural network used for message-passing features in the
        face-adjacency graph.

        Args:
            input_dim ([type]): [description]
            input_edge_dim ([type]): [description]
            output_dim ([type]): [description]
            hidden_dim (int, optional): [description]. Defaults to 64.
            learn_eps (bool, optional): [description]. Defaults to True.
            num_layers (int, optional): [description]. Defaults to 3.
            num_mlp_layers (int, optional): [description]. Defaults to 2.
        """
        super(UVNetGraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of layers for node and edge feature message passing
        self.node_conv_layers = torch.nn.ModuleList()
        self.edge_conv_layers = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            node_feats = input_dim if layer == 0 else hidden_dim
            edge_feats = input_edge_dim if layer == 0 else hidden_dim
            self.node_conv_layers.append(
                _NodeConv(
                    node_feats=node_feats,
                    out_feats=hidden_dim,
                    edge_feats=edge_feats,
                    num_mlp_layers=num_mlp_layers,
                    hidden_mlp_dim=hidden_dim,
                    conv_type=conv_type,
                ),
            )
            self.edge_conv_layers.append(
                _EdgeConv(
                    edge_feats=edge_feats,
                    out_feats=hidden_dim,
                    node_feats=node_feats,
                    num_mlp_layers=num_mlp_layers,
                    hidden_mlp_dim=hidden_dim,
                )
            )

        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        self.drop1 = nn.Dropout(0.3)
        self.drop = nn.Dropout(0.5)
        self.pool = MaxPooling()

    def forward(self, g, h, efeat):
        hidden_rep = [h]
        he = efeat

        for i in range(self.num_layers - 1):
            # Update node features
            h = self.node_conv_layers[i](g, h, he)
            # Update edge features
            he = self.edge_conv_layers[i](g, h, he)
            hidden_rep.append(h)

        out = hidden_rep[-1]
        out = self.drop1(out)
        score_over_layer = 0

        # Perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return out, score_over_layer
