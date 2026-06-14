#!/usr/bin/env python3
"""
5-fold cross-validation training of AttSiOff using ENsiRNA data.
"""
import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader

repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(repo_root, '..'))

from model import RNAFM_SIPRED_2
from load_data import (gibbs_energy, score_seq_by_pssm, secondary_struct,
                       get_tri_comp_percent, get_di_comp_percent,
                       get_single_comp_percent, get_gc_percentage, get_gc_sterch,
                       create_pssm)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f'Using device: {device}')

BATCH_SIZE = 128
LR = 0.005
N_EPOCHS = 500
PATIENCE = 20

class AttSiOffDataset(Dataset):
    def __init__(self, df, pssm, data_dir):
        self.df = df.reset_index(drop=True)
        self.pssm = pssm
        self.data_dir = data_dir
        self.sirna_dir = os.path.join(data_dir, 'RNAFM_sirna')
        self.mrna_dir = os.path.join(data_dir, 'RNAFM_mrna')
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['Antisense'].upper()
        rna_idx = f"{int(row['RNAFM_ind']):04d}"
        
        rnafm = np.load(os.path.join(self.sirna_dir, f'{rna_idx}.npy'), allow_pickle=True)
        rnafm_mrna = np.load(os.path.join(self.mrna_dir, f'{rna_idx}.npy'), allow_pickle=True)
        
        item = {
            'rnafm_encode': torch.tensor(rnafm, dtype=torch.float32),
            'rnafm_encode_mrna': torch.tensor(rnafm_mrna, dtype=torch.float32),
            'sirna_gibbs_energy': torch.tensor(gibbs_energy(seq[:19]), dtype=torch.float32),
            'pssm_score': torch.tensor(score_seq_by_pssm(self.pssm, seq), dtype=torch.float32),
            'gc_sterch': torch.tensor(get_gc_sterch(seq), dtype=torch.float32),
            'sirna_second_percent': torch.tensor(secondary_struct(seq)[0], dtype=torch.float32),
            'sirna_second_energy': torch.tensor(secondary_struct(seq)[1], dtype=torch.float32),
            'tri_nt_percent': torch.tensor(get_tri_comp_percent(seq), dtype=torch.float32),
            'di_nt_percent': torch.tensor(get_di_comp_percent(seq), dtype=torch.float32),
            'single_nt_percent': torch.tensor(get_single_comp_percent(seq), dtype=torch.float32),
            'gc_content': torch.tensor(get_gc_percentage(seq), dtype=torch.float32),
            'inhibit': torch.tensor(row['inhibition'], dtype=torch.float32),
        }
        return item

def collate_fn(batch):
    keys = ['rnafm_encode', 'rnafm_encode_mrna', 'sirna_gibbs_energy',
            'pssm_score', 'gc_sterch', 'sirna_second_percent',
            'sirna_second_energy', 'tri_nt_percent', 'di_nt_percent',
            'single_nt_percent', 'gc_content', 'inhibit']
    return {k: torch.stack([b[k] for b in batch]) for k in keys}

def train_fold(fold_k, data_dir):
    print(f'\n{"="*60}')
    print(f'Fold {fold_k}')
    print(f'{"="*60}')
    
    csv_path = os.path.join(data_dir, 'normalized_sirna_with_mrna.csv')
    all_data = pd.read_csv(csv_path)
    
    train_df = all_data[all_data['source_paper'] == 'train'].reset_index(drop=True)
    valid_df = all_data[all_data['source_paper'] == 'valid'].reset_index(drop=True)
    
    print(f'Train: {len(train_df)}, Valid: {len(valid_df)}')
    
    pssm_train = create_pssm(train_df['Antisense'].values)
    pssm_valid = create_pssm(valid_df['Antisense'].values)
    
    train_set = AttSiOffDataset(train_df, pssm_train, data_dir)
    valid_set = AttSiOffDataset(valid_df, pssm_valid, data_dir)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=len(valid_set), shuffle=False, drop_last=False, collate_fn=collate_fn)
    
    model = RNAFM_SIPRED_2(dp=0.1, device=device).to(torch.float32).to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=4)
    
    best_spcc = -1
    best_epoch = 0
    patience_counter = 0
    save_dir = os.path.join('./output', f'fold_{fold_k}')
    os.makedirs(save_dir, exist_ok=True)
    
    from scipy import stats
    
    for epoch in range(N_EPOCHS):
        model.train()
        for batch in train_loader:
            for k in batch.keys():
                batch[k] = batch[k].to(device).to(torch.float32)
            pred = model(batch)
            loss = criterion(pred, batch['inhibit'])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                for k in batch.keys():
                    batch[k] = batch[k].to(device).to(torch.float32)
                pred = model(batch)
                label = batch['inhibit']
                pred_np = pred.cpu().numpy().flatten()
                label_np = label.cpu().numpy().flatten()
                pcc = stats.pearsonr(pred_np, label_np)[0]
                spcc = stats.spearmanr(pred_np, label_np)[0]
        
        if spcc > best_spcc:
            best_spcc = spcc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth.tar'))
        else:
            patience_counter += 1
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f'Epoch {epoch+1:3d}: loss={loss.item():.4f}, valid PCC={pcc:.4f}, SPCC={spcc:.4f} (best={best_spcc:.4f}@{best_epoch})')
        
        if patience_counter >= PATIENCE:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth.tar'), map_location=device))
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            for k in batch.keys():
                batch[k] = batch[k].to(device).to(torch.float32)
            pred = model(batch)
            label = batch['inhibit']
            pred_np = pred.cpu().numpy().flatten()
            label_np = label.cpu().numpy().flatten()
            pcc = stats.pearsonr(pred_np, label_np)[0]
            spcc = stats.spearmanr(pred_np, label_np)[0]
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score((label_np > 0.7), pred_np)
    
    print(f'\nFold {fold_k} final: PCC={pcc:.4f}, SPCC={spcc:.4f}, AUC={auc:.4f}')
    return {'fold': fold_k, 'pcc': pcc, 'spcc': spcc, 'auc': auc}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default=os.path.join(repo_root, 'data'),
                        help='Root directory containing fold_{1-5}/ subdirectories')
    args = parser.parse_args()
    data_root = args.data_root
    
    results = []
    for k in range(1, 6):
        data_dir = os.path.join(data_root, f'fold_{k}')
        result = train_fold(k, data_dir)
        results.append(result)
    
    print('\n' + '='*60)
    print('5-Fold Cross-Validation Results')
    print('='*60)
    for r in results:
        print(f"Fold {r['fold']}: PCC={r['pcc']:.4f}, SPCC={r['spcc']:.4f}, AUC={r['auc']:.4f}")
    
    avg_pcc = np.mean([r['pcc'] for r in results])
    avg_spcc = np.mean([r['spcc'] for r in results])
    avg_auc = np.mean([r['auc'] for r in results])
    pcc_vals = [r['pcc'] for r in results]
    spcc_vals = [r['spcc'] for r in results]
    auc_vals = [r['auc'] for r in results]
    print(f'\nAverage: PCC={avg_pcc:.4f}±{np.std(pcc_vals):.4f}, '
          f'SPCC={avg_spcc:.4f}±{np.std(spcc_vals):.4f}, '
          f'AUC={avg_auc:.4f}±{np.std(auc_vals):.4f}')
