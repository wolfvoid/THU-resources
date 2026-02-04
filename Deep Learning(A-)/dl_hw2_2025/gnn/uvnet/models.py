import pytorch_lightning as pl
import torchmetrics
import torch
from torch import nn
import dgl
import torch.nn.functional as F
import uvnet.uvnet_encoders
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

class _NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        """
        A 3-layer MLP with linear outputs

        Args:
            input_dim (int): Dimension of the input tensor
            num_classes (int): Dimension of the output logits
            dropout (float, optional): Dropout used after each linear layer. Defaults to 0.3.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        """
        Forward pass

        Args:
            inp (torch.tensor): Inputs features to be mapped to logits
                                (batch_size x input_dim)

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

###############################################################################
# Segmentation model
###############################################################################
class UVNetSegmenter(nn.Module):
    """
    UV-Net solid face segmentation model
    """

    def __init__(
        self,
        num_classes,
        crv_in_channels=6,
        crv_emb_dim=64,
        srf_emb_dim=64,
        graph_emb_dim=128,
        dropout=0.3,
        conv_type='GCN',
    ):
        """
        Initialize the UV-Net solid face segmentation model

        Args:
            num_classes (int): Number of classes to output per-face
            crv_in_channels (int, optional): Number of input channels for the 1D edge UV-grids
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()
        # A 1D convolutional network to encode B-rep edge geometry represented as 1D UV-grids
        self.curv_encoder = uvnet.uvnet_encoders.UVNetCurveEncoder(
            in_channels=crv_in_channels, output_dims=crv_emb_dim
        )
        # A 2D convolutional network to encode B-rep face geometry represented as 2D UV-grids
        self.surf_encoder = uvnet.uvnet_encoders.UVNetSurfaceEncoder(
            in_channels=7, output_dims=srf_emb_dim
        )
        self.graph_encoder = uvnet.uvnet_encoders.UVNetGraphEncoder(
            srf_emb_dim, crv_emb_dim, graph_emb_dim,
            conv_type=conv_type,
        )
        # A non-linear classifier that maps face embeddings to face logits
        self.seg = _NonLinearClassifier(
            graph_emb_dim + srf_emb_dim, num_classes, dropout=dropout
        )

    def forward(self, batched_graph):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Logits (total_nodes_in_batch x num_classes)
        """
        # Input features
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        # Compute hidden edge and face features
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        # Message pass and compute per-face(node) and global embeddings
        node_emb, graph_emb = self.graph_encoder(
            batched_graph, hidden_srf_feat, hidden_crv_feat
        )
        # Repeat the global graph embedding so that it can be
        # concatenated to the per-node embeddings
        num_nodes_per_graph = batched_graph.batch_num_nodes().to(graph_emb.device)
        graph_emb = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
        local_global_feat = torch.cat((node_emb, graph_emb), dim=1)
        # Map to logits
        out = self.seg(local_global_feat)
        return out


class Segmentation(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the segmenter (per-face classifier).
    """

    def __init__(
        self,
        num_classes,
        graph_encoder_type,
        crv_in_channels=6,
        srf_emb_dim=64,
        crv_emb_dim=64,
        graph_emb_dim=128,
        dropout=0.3,
        batch_size=64,
        conv_type='GCN',
    ):
        """
        Args:
            num_classes (int): Number of per-face classes in the dataset
        """
        super().__init__()
        self.save_hyperparameters()

        if graph_encoder_type == "GNN":
            self.model = UVNetSegmenter(num_classes, crv_in_channels=crv_in_channels, conv_type=conv_type)
        else:
            raise ValueError(f"Unknown graph encoder type: {graph_encoder_type}")
        # Setting compute_on_step = False to compute "part IoU"
        # This is because we want to compute the IoU on the entire dataset
        # at the end to account for rare classes, rather than within each batch

        self.train_iou = MulticlassJaccardIndex(num_classes=num_classes)
        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes)
        self.test_iou = MulticlassJaccardIndex(num_classes=num_classes)

        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.batch_size=batch_size

    def forward(self, batched_graph):
        logits = self.model(batched_graph)
        return logits

    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = inputs.ndata["y"]
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        if torch.isnan(self.model.curv_encoder.state_dict()['conv1.0.weight']).any():
            import pdb; pdb.set_trace()
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        preds = F.softmax(logits, dim=-1)
        self.train_iou(preds, labels)
        self.train_accuracy(preds, labels)
        return loss

    def on_train_epoch_end(self):
        self.log("train_iou", self.train_iou.compute(), batch_size=self.batch_size)
        self.log("train_accuracy", self.train_accuracy.compute(), batch_size=self.batch_size)

    def validation_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = inputs.ndata["y"]
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        preds = F.softmax(logits, dim=-1)
        self.val_iou(preds, labels)
        self.val_accuracy(preds, labels)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_iou", self.val_iou.compute(), batch_size=self.batch_size)
        self.log("val_accuracy", self.val_accuracy.compute(), batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = inputs.ndata["y"]
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        preds = F.softmax(logits, dim=-1)
        self.test_iou(preds, labels)
        self.test_accuracy(preds, labels)

    def on_test_epoch_end(self):
        self.log("test_iou", self.test_iou.compute(), batch_size=self.batch_size)
        self.log("test_accuracy", self.test_accuracy.compute(), batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
