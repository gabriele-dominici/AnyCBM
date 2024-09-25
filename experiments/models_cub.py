from abc import abstractmethod
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss

class NeuralNet(pl.LightningModule):
    def __init__(self, input_features: int, n_classes: int, emb_size: int, learning_rate: float = 0.01):
        super().__init__()
        self.input_features = input_features
        self.n_classes = n_classes
        self.emb_size = emb_size
        self.learning_rate = learning_rate
        self.cross_entropy = CrossEntropyLoss(reduction="mean")
        self.bce = BCELoss(reduction="mean")
        self.bce_log = BCEWithLogitsLoss(reduction="mean")
        self.cross_entropy = CrossEntropyLoss(reduction="mean")
        self.nnl = torch.nn.NLLLoss(reduction="mean")

    @abstractmethod
    def forward(self, X):
        raise NotImplementedError

    @abstractmethod
    def _unpack_input(self, I):
        raise NotImplementedError

    def training_step(self, I, batch_idx):
        X, _, y_true = self._unpack_input(I)

        y_preds = self.forward(X)

        loss = self.cross_entropy(y_preds.squeeze(), y_true.float().argmax(dim=-1))
        task_accuracy = roc_auc_score(y_true, y_preds.detach())
        self.log("train_acc", task_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, I, batch_idx):
        X, _, y_true = self._unpack_input(I)
        y_preds = self.forward(X)
        loss = self.cross_entropy(y_preds.squeeze(), y_true.float().argmax(dim=-1))
        task_accuracy = roc_auc_score(y_true, y_preds.detach())
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", task_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, I, batch_idx):
        X, _, y_true = self._unpack_input(I)
        y_preds = self.forward(X)
        loss = self.cross_entropy(y_preds.squeeze(), y_true.float().argmax(dim=-1))
        task_accuracy = roc_auc_score(y_true, y_preds.detach())
        self.log("test_acc", task_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


class StandardE2E(NeuralNet):
    def __init__(self, input_features: int, n_classes: int, emb_size: int, learning_rate: float = 0.01):
        super().__init__(input_features, n_classes, emb_size, learning_rate)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_features, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
        )

    def _unpack_input(self, I):
        return I[0], I[1], I[2]

    def forward(self, X, explain=False):
        return self.decoder(self.encoder(X))

    def _forward_pre(self, X, explain=False):
        return self.encoder(X)

    def _forward_post(self, X, explain=False):
        return self.decoder(X)


class GenerativeCBM(NeuralNet):
    def __init__(self, input_features: int, n_concepts: int, emb_size: int,
                 learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 task_weight: float = 0.1):
        super().__init__(input_features, 0, emb_size, learning_rate)
        self.n_concepts = n_concepts
        self.concept_names = concept_names
        self.task_names = task_names
        self.task_weight = task_weight
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_features, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_concepts),
            torch.nn.Sigmoid()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_concepts, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, input_features),
            torch.nn.LeakyReLU()
        )
        self.mse = torch.nn.MSELoss()

    def _unpack_input(self, I):
        return I[0], I[1]

    def forward(self, X, explain=False):
        c_pred = self.encoder(X)
        # c_pred[:, :-2] = torch.softmax(c_pred[:, :-2], dim=1)
        # c_pred[:, -2:] = torch.softmax(c_pred[:, -2:], dim=1)
        return self.decoder(c_pred), c_pred

    def training_step(self, I, batch_idx):
        X, c_true = self._unpack_input(I)

        emb, c_preds = self.forward(X)

        concept_loss = self.bce(c_preds, c_true.float())
        # concept_loss_1 = self.nnl(torch.log(c_preds[:, :-2]), c_true[:, :-2].float().argmax(dim=-1))
        # concept_loss_2 = self.nnl(torch.log(c_preds[:, -2:]), c_true[:, -2:].float().argmax(dim=-1))
        mse_loss = self.mse(emb, X)
        loss = concept_loss + 0.1*mse_loss

        concept_accuracy = roc_auc_score(c_true, c_preds.detach())
        
        self.log("mse_loss", mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("concept_acc", concept_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, I, batch_idx):
        X, c_true = self._unpack_input(I)

        emb, c_preds = self.forward(X)

        concept_loss = self.bce(c_preds, c_true.float())
        # concept_loss_1 = self.nnl(torch.log(c_preds[:, :-2]), c_true[:, :-2].float().argmax(dim=-1))
        # concept_loss_2 = self.nnl(torch.log(c_preds[:, -2:]), c_true[:, -2:].float().argmax(dim=-1))
        mse_loss = self.mse(emb, X)
        loss = concept_loss + 0.1*mse_loss

        concept_accuracy = roc_auc_score(c_true, c_preds.detach())

        self.log("mse_loss", mse_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("concept_acc", concept_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    

    # def test_step(self, I, batch_idx):
    #     X, c_true, y_true = self._unpack_input(I)
    #
    #     c_preds, y_preds, _ = self.forward(X)
    #
    #     concept_loss = self.bce(c_preds, c_true.float())
    #     task_loss = self.bce_log(y_preds, y_true.float())
    #     loss = concept_loss + self.task_weight*task_loss
    #     task_accuracy = accuracy_score(y_true.squeeze(), y_preds.squeeze().detach()>0.5)
    #     concept_accuracy = accuracy_score(c_true, c_preds.squeeze().detach()>0.5)
    #
    #     self.log("test_acc", task_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("concept_acc", concept_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #
    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

class CBM(NeuralNet):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int,
                 learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 task_weight: float = 0.1):
        super().__init__(input_features, 0, emb_size, learning_rate)
        self.n_concepts = n_concepts
        self.concept_names = concept_names
        self.task_names = task_names
        self.task_weight = task_weight
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_features, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )
        self.concept_predictor = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_concepts),
            torch.nn.Sigmoid()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_concepts, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
        )

        self.mse = torch.nn.MSELoss()

    def _unpack_input(self, I):
        return I[0], I[1], I[2]

    def forward(self, X, explain=False):
        c_pred = self.concept_predictor(self.encoder(X))
        # c_pred[:, :-2] = torch.softmax(c_pred[:, :-2], dim=1)
        # c_pred[:, -2:] = torch.softmax(c_pred[:, -2:], dim=1)
        return self.decoder(c_pred), c_pred

    def training_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        y_pred, c_preds = self.forward(X)

        # concept_loss_1 = self.nnl(torch.log(c_preds[:, :-2]), c_true[:, :-2].float().argmax(dim=-1))
        # concept_loss_2 = self.nnl(torch.log(c_preds[:, -2:]), c_true[:, -2:].float().argmax(dim=-1))
        concept_loss = self.bce(c_preds, c_true.float())
        task_loss = self.bce_log(y_pred, y_true.float())
        loss = concept_loss + task_loss

        concept_accuracy = roc_auc_score(c_true, c_preds.detach())
        task_accuracy = roc_auc_score(y_true, y_pred.detach())

        self.log("task_acc", task_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("concept_acc", concept_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        y_pred, c_preds = self.forward(X)

        # concept_loss_1 = self.nnl(torch.log(c_preds[:, :-2]), c_true[:, :-2].float().argmax(dim=-1))
        # concept_loss_2 = self.nnl(torch.log(c_preds[:, -2:]), c_true[:, -2:].float().argmax(dim=-1))
        concept_loss = self.bce(c_preds, c_true.float())
        task_loss = self.bce_log(y_pred, y_true.float())
        loss = concept_loss + task_loss

        concept_accuracy = roc_auc_score(c_true, c_preds.detach())
        task_accuracy = roc_auc_score(y_true, y_pred.detach())
        
        self.log("task_acc", task_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("concept_acc", concept_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def test_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        y_pred, c_preds = self.forward(X)

        # concept_loss_1 = self.nnl(torch.log(c_preds[:, :-2]), c_true[:, :-2].float().argmax(dim=-1))
        # concept_loss_2 = self.nnl(torch.log(c_preds[:, -2:]), c_true[:, -2:].float().argmax(dim=-1))
        concept_loss = self.bce(c_preds, c_true.float())
        task_loss = self.bce_log(y_pred, y_true.float())
        loss = concept_loss + task_loss


        concept_accuracy = roc_auc_score(c_true, c_preds.detach())
        task_accuracy = roc_auc_score(y_true, y_pred.detach())
        
        self.log("task_acc", task_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("concept_acc", concept_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    

    # def test_step(self, I, batch_idx):
    #     X, c_true, y_true = self._unpack_input(I)
    #
    #     c_preds, y_preds, _ = self.forward(X)
    #
    #     concept_loss = self.bce(c_preds, c_true.float())
    #     task_loss = self.bce_log(y_preds, y_true.float())
    #     loss = concept_loss + self.task_weight*task_loss
    #     task_accuracy = accuracy_score(y_true.squeeze(), y_preds.squeeze().detach()>0.5)
    #     concept_accuracy = accuracy_score(c_true, c_preds.squeeze().detach()>0.5)
    #
    #     self.log("test_acc", task_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("concept_acc", concept_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #
    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
