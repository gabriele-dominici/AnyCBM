import torch
from matplotlib import pyplot as plt
import seaborn as sns
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import random

from experiments.colored_mnist.dataset import load_preprocessed_data
from experiments.colored_mnist.models import StandardE2E, GenerativeCBM, CBM

os.environ['CUDA_VISIBLE_DEVICES'] = ''
concept_family = {'Numbers': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]} 

x_train, c_train, y_train, x_test, c_test, y_test = load_preprocessed_data(base_dir='../../embeddings/mnist_colored/')
x_train_2, c_train_2, y_train_2, x_test_2, c_test_2, y_test_2 = load_preprocessed_data(base_dir='../../embeddings/mnist_bw/')
# take first half of x_train e sencode half of x_train_2
# x_train = x_train[:int(x_train.shape[0]/2)]
c_train = c_train[:, :-2]
# y_train = y_train[:int(y_train.shape[0]/2)]
# x_train_2 = x_train_2[int(x_train_2.shape[0]/2):]
c_train_2 = c_train_2[:, :-2]
# y_train_2 = y_train_2[int(y_train_2.shape[0]/2):]
# x_test = x_test[:int(x_test.shape[0]/2)]
c_test = c_test[:, :-2]
# y_test = y_test[:int(y_test.shape[0]/2)]
# x_test_2 = x_test_2[int(x_test_2.shape[0]/2):]
c_test_2 = c_test_2[:, :-2]
# y_test_2 = y_test_2[int(y_test_2.shape[0]/2):]
n_concepts = c_train.shape[1]
n_classes = y_train.shape[1]

train_dataset_2 = TensorDataset(x_train_2, c_train_2, y_train_2)
train_loader_2 = DataLoader(train_dataset_2, batch_size=200)
test_dataset_2 = TensorDataset(x_test_2, c_test_2, y_test_2)
test_loader_2 = DataLoader(test_dataset_2, batch_size=10000)
train_dataset = TensorDataset(x_train, c_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=200)
test_dataset = TensorDataset(x_test, c_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=10000)

results = {}
learning_rate = 0.01
emb_size = 32
epochs = 4
seed = [0,4,8,13,42]

for s in seed:
    models = [
    StandardE2E(x_train_2.shape[1], n_classes, emb_size=emb_size, learning_rate=learning_rate),
    CBM(x_train_2.shape[1], n_concepts, n_classes, emb_size=emb_size, learning_rate=learning_rate),
    ]
    generativeCBM = GenerativeCBM(emb_size, n_concepts, emb_size*2)
    results[s] = {}
    for model in models:
        results[s][model.__class__.__name__] = {}
        pl.seed_everything(s)
        
        if model.__class__.__name__ == 'StandardE2E':
            checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_weights_only=True)

            trainer = Trainer(max_epochs=epochs, accelerator='cpu', enable_checkpointing=True, callbacks=checkpoint_callback)
            trainer.fit(model, train_loader_2, test_loader_2)
            model.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])
            model.eval()
            trainer.test(model, test_loader_2)
            y_preds_bb = model(x_test_2)
            task_accuracy_bb = roc_auc_score(y_test_2, y_preds_bb.detach())
            print(f'Accuracy - task: {task_accuracy_bb:.4f}')
            y_preds_bb = model(x_test)
            task_accuracy_bb_2 = roc_auc_score(y_test, y_preds_bb.detach())
            print(f'Accuracy - task: {task_accuracy_bb_2:.4f}')
            emb_train = model.encoder(x_train)
            emb_test = model.encoder(x_test)
            emb_train_loader = DataLoader(TensorDataset(emb_train.detach(), c_train), batch_size=200)
            emb_test_loader = DataLoader(TensorDataset(emb_test.detach(), c_test), batch_size=10000)
            

            checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_weights_only=True)
            trainer2 = Trainer(max_epochs=30, accelerator='cpu', enable_checkpointing=True, callbacks=checkpoint_callback)
            trainer2.fit(generativeCBM, emb_train_loader, emb_test_loader)
            generativeCBM.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])
            generativeCBM.eval()
            
            emb_test = model.encoder(x_test_2)
            emb_cbm, c_preds = generativeCBM(emb_test)
            y_preds = model.decoder(emb_cbm)

            task_accuracy = roc_auc_score(y_test_2, y_preds.detach())
            concept_accuracy = roc_auc_score(c_test_2, c_preds.detach())
            print(f'Accuracy - task: {task_accuracy:.4f} concept: {concept_accuracy:.4f}')
            for i in range(c_test_2.shape[1]):
                print(f'Concept {i} - {roc_auc_score(c_test_2[:, i].detach().numpy(), c_preds[:, i].detach().numpy()):.4f}')
            family_keys = list(concept_family.keys())
            # randomize the order of the keys
            random.shuffle(family_keys)
            concepts_to_intervene = []
            percentile_99 = torch.quantile(c_preds, 0.99, dim=0)
            percentile_1 = torch.quantile(c_preds, 0.01, dim=0)
            c_test_interv_pos = (c_test_2 == 1) * percentile_99
            c_test_interv_neg = (c_test_2 == 0) * percentile_1
            c_test_interv = c_test_interv_pos + c_test_interv_neg
            noise_embs = emb_test.clone() + torch.randn_like(emb_test) * 20
            y_preds_noise = model.decoder(noise_embs)
            acc_int = roc_auc_score(y_test_2, y_preds_noise.detach())
            int_accuracy = [acc_int]
            _, c_preds_noise = generativeCBM(noise_embs)
            for key in family_keys:
                concepts_to_intervene += concept_family[key]
                concepts_to_intervene = sorted(concepts_to_intervene, reverse=False) 
                c_noise_int = c_preds_noise.clone()
                c_noise_int[:, torch.tensor(concepts_to_intervene).squeeze()] = c_test_interv[:, torch.tensor(concepts_to_intervene).squeeze()]
                y_preds_interv = model.decoder(generativeCBM.decoder(c_noise_int))
                acc_int = roc_auc_score(y_test_2, y_preds_interv.detach())
                int_accuracy.append(acc_int)
            print(int_accuracy)
            # interventions
            noise_embs = emb_test.clone() + torch.randn_like(emb_test) * 20
            y_preds_noise = model.decoder(noise_embs)
            y_preds_interv = model.decoder(generativeCBM.decoder(c_test_interv))
            all_fives = torch.zeros_like(c_test)
            all_fives[:, 5] = 1
            all_fives[:, -1] = 1
            y_fives = model.decoder(generativeCBM.decoder(all_fives))

            task_accuracy_noise = roc_auc_score(y_test_2, y_preds_noise.detach())
            task_accuracy_interv = roc_auc_score(y_test_2, y_preds_interv.detach())
            task_accuracy_fives = accuracy_score(torch.zeros_like(y_test), y_fives > 0)
            print(f'Accuracy - task noise: {task_accuracy_noise:.4f} task interv: {task_accuracy_interv:.4f}')
            print(f'Accuracy - task fives: {task_accuracy_fives:.4f}')
            results[s][model.__class__.__name__] = (task_accuracy_bb, task_accuracy, concept_accuracy, task_accuracy_noise, task_accuracy_interv, acc_int)
        else:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_weights_only=True)

            trainer = Trainer(max_epochs=epochs, accelerator='cpu', enable_checkpointing=True, callbacks=checkpoint_callback)
            trainer.fit(model, train_loader, test_loader)
            model.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])
            model.eval()
            trainer.test(model, test_loader)

            y_preds, c_preds = model(x_test)

            task_accuracy = roc_auc_score(y_test, y_preds.detach())
            concept_accuracy = roc_auc_score(c_test, c_preds.detach())
            print(f'Accuracy - task: {task_accuracy:.4f} concept: {concept_accuracy:.4f}')

            # interventions
            # interventions
            family_keys = list(concept_family.keys())
            # randomize the order of the keys
            concepts_to_intervene = []
            percentile_99 = torch.quantile(c_preds, 0.99, dim=0)
            percentile_1 = torch.quantile(c_preds, 0.01, dim=0)
            c_test_interv_pos = (c_test == 1) * percentile_99
            c_test_interv_neg = (c_test == 0) * percentile_1
            c_test_interv = c_test_interv_pos + c_test_interv_neg
            emb_test = model.encoder(x_test)
            noise_embs = emb_test.clone() + torch.randn_like(emb_test) * 20
            y_preds_noise = model.decoder(model.concept_predictor(noise_embs))
            acc_int = roc_auc_score(y_test, y_preds_noise.detach())
            int_accuracy = [acc_int]
            c_preds_noise = model.concept_predictor(noise_embs)
            for key in family_keys:
                concepts_to_intervene += concept_family[key]
                concepts_to_intervene = sorted(concepts_to_intervene, reverse=False) 
                c_noise_int = c_preds_noise.clone()
                c_noise_int[:, torch.tensor(concepts_to_intervene).squeeze()] = c_test_interv[:, torch.tensor(concepts_to_intervene).squeeze()]
                y_preds_interv = model.decoder(c_noise_int)
                acc_int = roc_auc_score(y_test, y_preds_interv.detach())
                int_accuracy.append(acc_int)
            print(int_accuracy)
            all_fives = torch.zeros_like(c_test)
            all_fives[:, 5] = 1
            all_fives[:, -1] = 1
            y_fives = model.decoder(all_fives)

            task_accuracy_noise = roc_auc_score(y_test, y_preds_noise.detach())
            task_accuracy_interv = roc_auc_score(y_test, y_preds_interv.detach())
            task_accuracy_fives = accuracy_score(torch.zeros_like(y_test), y_fives > 0.5)
            print(f'Accuracy - task noise: {task_accuracy_noise:.4f} task interv: {task_accuracy_interv:.4f}')
            print(f'Accuracy - task fives: {task_accuracy_fives:.4f}')
            results[s][model.__class__.__name__] = (task_accuracy, concept_accuracy, task_accuracy_noise, task_accuracy_interv, acc_int)
# save results as json
import json
with open('results_2d.json', 'w') as f:
    json.dump(results, f)