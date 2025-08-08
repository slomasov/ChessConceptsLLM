import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch.utils.data as data
from tqdm import tqdm


import torch
import random
import numpy as np
import pandas as pd

import torch.nn.init as init


import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import copy

from types import SimpleNamespace

from sklearn.model_selection import KFold, train_test_split

import argparse

class DatasetStaticLoad(torch.utils.data.Dataset):
    def __init__(self, csv_file_path, concept_name=None, all_sequence=False,
                    in_local=False, device="cuda", max_pos_samples=300):
        # accept *either* a single path, a list of paths, or DataFrame(s)
        if isinstance(csv_file_path, (list, tuple)):
            dfs = [(pd.read_csv(p) if isinstance(p, str) else p)
                      for p in csv_file_path]
            self.data = pd.concat(dfs, ignore_index=True)
        elif isinstance(csv_file_path, pd.DataFrame):
            self.data = csv_file_path.copy()
        else:
            self.data = pd.read_csv(csv_file_path)
        self.csv_file_path = csv_file_path
        self.concept_name = concept_name
        self.columns = self.data.columns.tolist()
        self.column_indices = {column: idx for idx, column in enumerate(self.columns)}
        self.max_pos_samples = max_pos_samples
        self.positive_row_idx = self.filter_by_concept()
        self.negative_rows_idx = self.filter_non_concepts()
        random.shuffle(self.negative_rows_idx)
        self.all_sequence = all_sequence
        self.in_local = in_local
        self.device = device  # Device to preload data
        self.mode = 'train'  # Default mode
        self.load_activations()
        if len(self.positive_row_idx) == 0 or len(self.negative_rows_idx) == 0:
            raise ValueError(
                f"Dataset for concept '{self.concept_name}' needs at least one "
                f"positive and one negative row, but got "
                f"{len(self.positive_row_idx)} positive / "
                f"{len(self.negative_rows_idx)} negative.")

    def load_activations(self):
        self.activations = None
        past_activations = []

        for i, row in self.data.iterrows():
            activation_file = row['Layer Outputs File']
            #if self.in_local: # An optimization for running in the google colab environment (you can ignore)
            #    activation_file = activation_file.replace('/content/drive/Shareddrives/CS229/searchless_chess/data/concept_data/npys/', '/content/')
            #replace activation file with the path to the activations
            activation_file = activation_file.replace('/content/drive/Shareddrives/CS229/', '/home/semyonlomasov/cs229_drive/')
            activation_idx = int(row['Layer Outputs Index'])
            activations = torch.from_numpy(np.load(activation_file))
            if len(activations.shape) == 3:
                activations = activations.unsqueeze(-1)
            if not self.all_sequence:
                activations = activations[:, :, -1, :]  # Get the CLS token rep.
            past_activations.append(activations[activation_idx])
            if len(past_activations) == 50 or i == len(self.data) - 1:
                batch_activations = torch.stack(past_activations).cpu()  # Move batch to GPU
                if self.activations is None:
                    self.activations = batch_activations
                else:
                    self.activations = torch.cat([self.activations, batch_activations], dim=0)
                past_activations = []

    def set_mode(self, mode):
        self.mode = mode

    def filter_by_concept(self):
        filtered_rows = self.data[self.data[self.concept_name] == 1].index.tolist()
        if len(filtered_rows) > self.max_pos_samples:
            filtered_rows = random.sample(filtered_rows, self.max_pos_samples)
        return filtered_rows

    def filter_non_concepts(self):
        filtered_rows = self.data[self.data[self.concept_name] == 0].index.tolist()
        return filtered_rows

    def __len__(self):
        return len(self.positive_row_idx)

    def get_item(self, idx):
        return self.activations[idx]

    def __getitem__(self, idx):
        if self.mode == 'train':
            neg_idx = random.choice(self.negative_rows_idx)
        else:
            neg_idx = self.negative_rows_idx[idx]
        pos_idx = self.positive_row_idx[idx]
        neg_item = self.get_item(neg_idx).to(self.device)
        pos_item = self.get_item(pos_idx).to(self.device)
        pos_item = pos_item.float()
        neg_item = neg_item.float()
        return pos_item, neg_item
    
    def cleanup(self):
        """Free GPU memory by clearing the activations tensor."""
        if self.activations is not None:
            del self.activations
            self.activations = None
        torch.cuda.empty_cache()  # Free any unused cached memory

class MinConceptVector(nn.Module):
    def __init__(self, feature_dim):
        super(MinConceptVector, self).__init__()
        self.concept_vector = nn.Parameter(torch.empty(feature_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Applying Xavier Initialization for general usage
        init.xavier_normal_(self.concept_vector.unsqueeze(0))

    def forward(self, x):
        # x is of shape: [B, E] (E --> feature_dim)
        return x @ self.concept_vector


class LogisticRegression(nn.Module):
    def __init__(self, feature_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(feature_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        # Applying Xavier initialization for the weights
        init.xavier_normal_(self.linear.weight)
        # Set bias to a small constant for stability
        init.constant_(self.linear.bias, 0)

    def forward(self, x):
        # x is of shape: [B, E] (E --> feature_dim)
        return torch.sigmoid(self.linear(x))

class MLPLogistic(nn.Module):
    def __init__(self, n_layers, feature_dim, hidden_dim, output_dim):
        super(MLPLogistic, self).__init__()
        self.n_layers = n_layers
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(feature_dim, hidden_dim))
        for i in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            init.xavier_normal_(layer.weight)
            init.constant_(layer.bias, 0)

    def forward(self, x):
        # x is of shape: [B, E] (E --> feature_dim)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.n_layers - 1:
                x = torch.relu(x)
        return torch.sigmoid(x)

class AllSeqNN(nn.Module):
    def __init__(self, seq_len, feature_dim, seq_first=True):
        super(AllSeqNN, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.seq_first = seq_first
        if self.seq_first:
            self.linear1 = nn.Linear(seq_len, 1)
            self.linear2 = nn.Linear(feature_dim, 1)
        else:
            self.linear1 = nn.Linear(feature_dim, 1)
            self.linear2 = nn.Linear(seq_len, 1)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal_(self.linear1.weight)
        init.constant_(self.linear1.bias, 0)
        init.xavier_normal_(self.linear2.weight)
        init.constant_(self.linear2.bias, 0)

    def forward(self, x):
        # x is of shape: [B, S, E] (S --> seq_len, E --> feature_dim)
        if self.seq_first:
            x = x.transpose(1, 2) # [B, E, S]
        x = self.linear1(x) # either [B, E, 1] or [B, S, 1]
        x = x.squeeze(-1) # [B, E] or [B, S]
        x = torch.relu(x)
        x = self.linear2(x) # [B, 1]
        x = x.squeeze(-1) # [B]
        return torch.sigmoid(x)


def evaluate_model(model, dataloader, config):
    model.eval()

    config.dataset.set_mode('eval')

    total_constraints = 0
    total_constraints_satisfied = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch_pos, batch_neg in dataloader:

            if config.layer_idx is not None:
                batch_pos = batch_pos[:, config.layer_idx, :]
                batch_neg = batch_neg[:, config.layer_idx, :]

            batch_pos = batch_pos.to(config.device)
            batch_neg = batch_neg.to(config.device)

            pos_scores = model(batch_pos)
            neg_scores = model(batch_neg)

            l1_lambda = config.l1_lambda if hasattr(config, 'l1_lambda') else 0.0
            l1_loss = l1_lambda * sum(torch.abs(param).sum() for param in model.parameters())

            # Constraint satisfaction check
            if config.model_class == MinConceptVector:
                satisfied_constraints = (pos_scores > neg_scores + config.margin).float().sum().item()
                total_constraints += len(pos_scores)
                total_constraints_satisfied += satisfied_constraints
                constraint_loss = torch.relu(config.margin - (pos_scores - neg_scores)).sum()

            elif config.model_class == LogisticRegression or config.model_class == MLPLogistic or config.model_class == AllSeqNN:
                satisfied_constraints = (pos_scores > 0.5).float().sum().item() + (neg_scores < 0.5).float().sum().item()
                total_constraints += len(pos_scores) + len(neg_scores)
                total_constraints_satisfied += satisfied_constraints
                constraint_loss = ((\
                    F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores)) + \
                    F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))) / 2).sum()

            else:
                raise ValueError("Invalid model name")

            total_loss += constraint_loss.item() + l1_loss.item()


    avg_constraint_satisfaction = total_constraints_satisfied / total_constraints if total_constraints > 0 else 0
    avg_loss = total_loss / total_constraints if total_constraints > 0 else 0

    return avg_constraint_satisfaction, avg_loss

def train_model(model, train_dataloader, val_dataloader, config):
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    best_model_state = None
    best_val_accuracy = -float('inf')
    best_val_loss = float('inf')

    recent_val_loss = None
    recent_val_accuracy = None

    cnt_buffer_epochs = 0

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        if config.verbose:
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        else:
            progress_bar = train_dataloader

        model.train()
        config.dataset.set_mode('train')
        for batch_pos, batch_neg in progress_bar:
            if config.layer_idx is not None:
                batch_pos = batch_pos[:, config.layer_idx, :]
                batch_neg = batch_neg[:, config.layer_idx, :]

            batch_pos = batch_pos.to(config.device)
            batch_neg = batch_neg.to(config.device)

            optimizer.zero_grad()
            pos_scores = model(batch_pos)
            neg_scores = model(batch_neg)

            if config.model_class == MinConceptVector:
                constraint_loss = torch.relu(config.margin - (pos_scores - neg_scores)).mean()
            elif config.model_class == LogisticRegression or config.model_class == MLPLogistic or config.model_class == AllSeqNN:
                constraint_loss = ((
                    F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores)) +
                    F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))) / 2).mean()
            else:
                raise ValueError("Invalid model name")

            l1_lambda = config.l1_lambda if hasattr(config, 'l1_lambda') else 0.0
            l1_loss = l1_lambda * sum(torch.abs(param).sum() for param in model.parameters())

            loss = constraint_loss + l1_loss
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            epoch_loss += loss.item() * batch_pos.size(0)

            # Calculate accuracy
            predicted_pos = (pos_scores > 0.5).float()
            predicted_neg = (neg_scores < 0.5).float()
            correct_predictions += (predicted_pos.sum() + predicted_neg.sum()).item()
            total_predictions += batch_pos.size(0) * 2

            # Update progress bar
            avg_loss = epoch_loss / total_predictions
            accuracy = correct_predictions / total_predictions

            if config.verbose:
                progress_bar.set_postfix({'loss': avg_loss, 'accuracy': accuracy, 'val_loss': recent_val_loss, 'val_accuracy': recent_val_accuracy, 'best_val_accuracy': best_val_accuracy, 'best_val_loss': best_val_loss})

        cnt_buffer_epochs += 1
        if (epoch + 1) % config.evals_per_epoch == 0 or epoch == config.num_epochs - 1:
            recent_val_accuracy, recent_val_loss = evaluate_model(model, val_dataloader, config)
            if recent_val_accuracy > best_val_accuracy:
                best_val_accuracy = recent_val_accuracy
                best_val_loss = recent_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
            if (recent_val_accuracy - 0.5) * 100 < cnt_buffer_epochs:

                if config.verbose:
                    print('The model is not improving, reset the model params')

                cnt_buffer_epochs = 0
                model.reset_parameters()
                optimizer = optim.Adam(model.parameters(), lr=config.lr)
                model.to(config.device)

    return best_model_state, best_val_accuracy, best_val_loss


def k_fold_cross_validation(config, external_test_dataset=None):
    kfold = KFold(n_splits=config.k, shuffle=True, random_state=42)
    avg_constraint_satisfactions = []
    avg_losses = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(config.dataset)):

        if config.verbose:
            print(f"Fold {fold+1}/{config.k}")

        # Split dataset into train and test subsets
        train_val_indices = train_idx
        test_subset_internal = data.Subset(config.dataset, test_idx)

        # Split train_val into train and validation sets
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=1.0 / (config.k - 1), random_state=42
        )
        train_subset = data.Subset(config.dataset, train_indices)
        val_subset = data.Subset(config.dataset, val_indices)

        # DataLoaders for train, validation, and test subsets
        train_loader = data.DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
        val_loader = data.DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)

        test_loader = (data.DataLoader(external_test_dataset,
                                                batch_size=config.batch_size,
                                                shuffle=False)
                       if external_test_dataset is not None
                       else data.DataLoader(test_subset_internal,
                                            batch_size=config.batch_size,
                                            shuffle=False))

        # Initialize model based on configuration
        model = config.model_class(**vars(config.model_params))
        # to cuda
        model.to(config.device)

        # Train model on training set for the current fold, with validation
        model_state, _, _ = train_model(
            model, train_loader, val_loader, config
        )

        # test in the test set
        model.load_state_dict(model_state)
        model.to(config.device)

        eval_constraint_satisfaction, eval_corresponding_loss = evaluate_model(
            model, val_loader, config
        )
        if config.verbose:
            print(f'In eval, we achieve {eval_constraint_satisfaction} accuracy, with loss {eval_corresponding_loss}')

        test_constraint_satisfaction, test_corresponding_loss = evaluate_model(
            model, test_loader, config
        )

        if config.verbose:
            print(f'In test, we achieve {test_constraint_satisfaction} accuracy, with loss {test_corresponding_loss}')

        # Collect fold results
        avg_constraint_satisfactions.append(test_constraint_satisfaction)
        avg_losses.append(test_corresponding_loss)

    # Average metrics across all folds
    overall_constraint_satisfaction = np.mean(avg_constraint_satisfactions)
    overall_loss = np.mean(avg_losses)
    return overall_constraint_satisfaction, overall_loss

def eval_concept(config):
    # ── assemble DataFrames for train / test ───────────────────────────
    train_set = set(config.train_csv_paths)
    test_set  = set(config.test_csv_paths)

    train_dfs, test_dfs = [], []

    # ❶ CSVs that appear in train list
    for p in train_set:
        df = pd.read_csv(p)
        if p in test_set:
            df_train, df_test = train_test_split(
                df,
                test_size=config.redundant_train_holdout_frac,
                random_state=42,
                shuffle=True
            )
            train_dfs.append(df_train)
            test_dfs.append(df_test)
        else:
            train_dfs.append(df)

    # ❷ CSVs that are only in the test list
    for p in (test_set - train_set):
        test_dfs.append(pd.read_csv(p))

    # ── create Dataset objects ─────────────────────────────────────────
    config.dataset = config.dataset_class(
        train_dfs,
        config.concept_name,
        config.all_sequence,
        device=config.device
    )

    external_test_ds = None
    if test_dfs:  # build only if we actually have held-out rows
        external_test_ds = config.dataset_class(
            test_dfs,
            config.concept_name,
            config.all_sequence,
            device=config.device
        )

    # ── run CV (potentially multiple independent repetitions) ─────────
    accs, losses = [], []
    for _ in range(config.num_its):
        acc, loss = k_fold_cross_validation(config, external_test_ds)
        accs.append(acc)
        losses.append(loss)

    # ── cleanup GPU memory ─────────────────────────────────────────────
    config.dataset.cleanup()
    if external_test_ds:
        external_test_ds.cleanup()
    torch.cuda.empty_cache()

    return float(np.mean(accs)), float(np.mean(losses))


def dict_to_namespace(d):
    return SimpleNamespace(
        **{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()}
    )

config = {
    'csv_file_path': None,
    'concept_name': None,
    'layer_idx': None,
    'num_its': 3,
    'num_epochs': 100,
    'margin': 0,
    'k': 6,
    'lr': 0.003,
    'l1_lambda': 0.01,
    'dataset_class': DatasetStaticLoad,
    'batch_size': 8,
    'model_class': None,
    'evals_per_epoch': 1,
    'all_sequence': None,
    'device': 'cuda',
    'model_params': None,
    'verbose': False,
}

parser = argparse.ArgumentParser()
parser.add_argument("--train_csv_paths", nargs='+', required=True,
                    help="CSV(s) mixed for training / validation")
parser.add_argument("--test_csv_paths", nargs='*', default=[],
                    help="CSV(s) reserved for final testing")
parser.add_argument("--redundant_train_holdout_frac", type=float, default=0.5,
                    help="If a CSV appears in both train & test, "
                         "this fraction is held out for testing")
parser.add_argument("--concept_name", type=str, default=None)
parser.add_argument("--layer_idx", type=int, default=None)
parser.add_argument("--seq_type", type=str, default="input")
parser.add_argument("--model_idx", type=int, default=0)

args = parser.parse_args()
config['train_csv_paths'] = args.train_csv_paths
config['test_csv_paths'] = args.test_csv_paths
config['redundant_train_holdout_frac'] = args.redundant_train_holdout_frac
config['concept_name'] = args.concept_name
config['layer_idx'] = args.layer_idx

models_layer = [
    (LogisticRegression, {'feature_dim': 1024}, False),
    (MinConceptVector, {'feature_dim': 1024}, False),
    (AllSeqNN, {'seq_len': 79, 'feature_dim': 1024, 'seq_first': True}, True),
]

models_input = [
    (LogisticRegression, {'feature_dim': 1}, False),
    (MinConceptVector, {'feature_dim': 1}, False),
    (AllSeqNN, {'seq_len': 79, 'feature_dim': 1, 'seq_first': True}, True),
]

if args.seq_type == 'input':
    model = models_input[args.model_idx]
else:
    model = models_layer[args.model_idx] 

config['model_class'] = model[0]
config['model_params'] = model[1]
config['all_sequence'] = model[2]

config = dict_to_namespace(config)

overall_constraint_satisfaction, overall_loss = eval_concept(
                    config
                )

print(f'we achieve {overall_constraint_satisfaction} accuracy, with loss {overall_loss}')