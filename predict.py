#!/usr/bin/env python3
"""
predict.py — AttSiOff 推理脚本
使用训练好的 RNAFM_SIPRED_2 模型对 siRNA-mRNA 对进行抑制效率预测。

用法:
  1) 使用预计算的 RNA-FM embedding (推荐)
     python predict.py --checkpoint model.pth.tar \\
         --sirna_emb sirna_001.npy --mrna_emb mrna_001.npy --pssm pssm.npy

  2) 输入序列 + 自动 RNA-FM 编码
     python predict.py --checkpoint model.pth.tar \\
         --sirna AAGGUUGGGCUGGUGUAUUAA --mrna 59-nt-window --pssm pssm.npy

  3) CSV 批量预测
     python predict.py --checkpoint model.pth.tar \\
         --csv input.csv --pssm pssm.npy

  4) 使用预计算 embedding 批量预测
     python predict.py --checkpoint model.pth.tar \\
         --csv input.csv --emb_dir ./embeddings \\
         --sirna_emb_col sirna_emb --mrna_emb_col mrna_emb

CSV 必须包含列:
  - 序列模式: 'sirna', 'mrna' (mrna 为 59 nt 窗口, 含 '.' 填充)
  - embedding 模式: 指向 .npy 文件的列名 (默认 'sirna_emb', 'mrna_emb')
  可选: 'pssm' 列 (指向 .npy 文件), 如未提供则使用 --pssm 参数
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')


RNAFM_AVAILABLE = False


def _lazy_imports():
    global np, pd, torch, RNAFM_SIPRED_2
    global gibbs_energy, score_seq_by_pssm, secondary_struct
    global get_tri_comp_percent, get_di_comp_percent
    global get_single_comp_percent, get_gc_percentage, get_gc_sterch
    global create_pssm

    import numpy as np
    import pandas as pd
    import torch

    repo_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, repo_root)

    from model import RNAFM_SIPRED_2
    from load_data import (
        gibbs_energy, score_seq_by_pssm, secondary_struct,
        get_tri_comp_percent, get_di_comp_percent,
        get_single_comp_percent, get_gc_percentage, get_gc_sterch,
        create_pssm,
    )

    global RNAFM_AVAILABLE
    try:
        import fm
        RNAFM_AVAILABLE = True
    except ImportError:
        RNAFM_AVAILABLE = False


def get_device():
    _lazy_imports()
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, device, dp=0.1):
    _lazy_imports()
    model = RNAFM_SIPRED_2(dp=dp, device=device).to(torch.float32).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def compute_hand_features(sirna_seq, pssm=None):
    _lazy_imports()
    seq = sirna_seq.upper().replace('T', 'U')
    features = {}
    features['sirna_gibbs_energy'] = gibbs_energy(seq[:19]).astype(np.float32)
    if pssm is not None:
        features['pssm_score'] = score_seq_by_pssm(pssm, seq).astype(np.float32)
    else:
        features['pssm_score'] = np.array([[0.0]], dtype=np.float32)
    features['gc_sterch'] = get_gc_sterch(seq).astype(np.float32)
    sec_pct, sec_energy = secondary_struct(seq)
    features['sirna_second_percent'] = sec_pct.astype(np.float32)
    features['sirna_second_energy'] = sec_energy.astype(np.float32)
    features['tri_nt_percent'] = get_tri_comp_percent(seq).astype(np.float32)
    features['di_nt_percent'] = get_di_comp_percent(seq).astype(np.float32)
    features['single_nt_percent'] = get_single_comp_percent(seq).astype(np.float32)
    features['gc_content'] = get_gc_percentage(seq).astype(np.float32)
    return features


def load_rnafm_model(device):
    _lazy_imports()
    if not RNAFM_AVAILABLE:
        raise ImportError(
            "rna-fm 未安装, 无法自动计算 embedding。\n"
            "请安装: pip install rna-fm\n"
            "或使用预计算的 .npy embedding 文件。"
        )
    import fm
    model, alphabet = fm.pretrained.rna_fm_t12()
    model.eval()
    model.to(device)
    return model, alphabet


def compute_rnafm_embedding(seq, rnafm_model, alphabet, device):
    _lazy_imports()
    seq = seq.upper().replace('T', 'U')
    _, _, tokens = alphabet.get_batch_converter()([("seq", seq)])
    tokens = tokens.to(device)
    with torch.no_grad():
        results = rnafm_model(tokens, repr_layers=[12])
        emb = results["representations"][12][0, 1:-1].cpu().numpy().astype(np.float32)
    return emb


def extract_mrna_window(mrna_seq, position, anti_len=19, window_len=59):
    pad_len = (window_len - anti_len) // 2
    start = position - pad_len
    end = position + anti_len + pad_len
    if start < 0:
        left_pad = "." * (-start)
        window = left_pad + mrna_seq[:end]
    elif end > len(mrna_seq):
        right_pad = "." * (end - len(mrna_seq))
        window = mrna_seq[start:] + right_pad
    else:
        window = mrna_seq[start:end]
    return window.upper()


def collate_to_batch(items, device):
    _lazy_imports()
    batch = {}
    keys = [
        'rnafm_encode', 'rnafm_encode_mrna',
        'sirna_gibbs_energy', 'pssm_score', 'gc_sterch',
        'sirna_second_percent', 'sirna_second_energy',
        'tri_nt_percent', 'di_nt_percent',
        'single_nt_percent', 'gc_content',
    ]
    for k in keys:
        tensors = [torch.tensor(item[k], dtype=torch.float32) for item in items]
        batch[k] = torch.stack(tensors).to(device)
    return batch


def predict_batch(model, items, device):
    _lazy_imports()
    batch = collate_to_batch(items, device)
    with torch.no_grad():
        pred = model(batch)
    return pred.cpu().numpy()


def run_predictions(model, device, sequences, pssm=None,
                    rnafm_model=None, alphabet=None,
                    emb_dir=None, sirna_emb_col=None, mrna_emb_col=None,
                    verbose=True):
    _lazy_imports()
    items = []
    for i, row in enumerate(sequences):
        item = {}

        if sirna_emb_col and mrna_emb_col:
            npy_dir = emb_dir or "."
            sirna_path = row[sirna_emb_col]
            mrna_path = row[mrna_emb_col]
            if not os.path.isabs(sirna_path):
                sirna_path = os.path.join(npy_dir, sirna_path)
            if not os.path.isabs(mrna_path):
                mrna_path = os.path.join(npy_dir, mrna_path)
            item['rnafm_encode'] = np.load(sirna_path, allow_pickle=True).astype(np.float32)
            item['rnafm_encode_mrna'] = np.load(mrna_path, allow_pickle=True).astype(np.float32)
        else:
            sirna_seq = row['sirna'].upper().replace('T', 'U')
            mrna_seq = row['mrna'].upper().replace('T', 'U')
            if len(sirna_seq) < 21:
                sirna_seq = sirna_seq + 'AA'
            item['rnafm_encode'] = compute_rnafm_embedding(sirna_seq, rnafm_model, alphabet, device)
            item['rnafm_encode_mrna'] = compute_rnafm_embedding(mrna_seq, rnafm_model, alphabet, device)

        hand = compute_hand_features(row['sirna'], pssm)
        item.update(hand)
        items.append(item)

        if verbose and (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(sequences)}")

    preds = predict_batch(model, items, device)
    return preds


def main():
    parser = argparse.ArgumentParser(
        description='AttSiOff 推理脚本 — 使用训练好的模型预测 siRNA 抑制效率'
    )
    parser.add_argument('--checkpoint', '-c', required=True,
                        help='训练好的模型权重文件 (.pth.tar)')
    parser.add_argument('--pssm', type=str, default=None,
                        help='PSSM 矩阵文件 (.npy), 如未提供则由序列自动计算')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sirna', type=str, default=None,
                       help='单条 siRNA 序列 (21 nt)')
    group.add_argument('--csv', type=str, default=None,
                       help='批量预测的 CSV 文件路径')

    parser.add_argument('--mrna', type=str, default=None,
                        help='单条模式下的 mRNA 59 nt 窗口序列')
    parser.add_argument('--position', type=int, default=None,
                        help='siRNA 在 mRNA 上的结合位置 (与 --mrna_full 配合使用)')
    parser.add_argument('--mrna_full', type=str, default=None,
                        help='完整的 mRNA 序列 (需配合 --position)')

    parser.add_argument('--emb_dir', type=str, default=None,
                        help='预计算 embedding .npy 文件的目录 (CSV 模式)')
    parser.add_argument('--sirna_emb', type=str, default=None,
                        help='单条模式下 siRNA embedding .npy 文件路径')
    parser.add_argument('--mrna_emb', type=str, default=None,
                        help='单条模式下 mRNA embedding .npy 文件路径')
    parser.add_argument('--sirna_emb_col', type=str, default='sirna_emb',
                        help='CSV 中 siRNA embedding 列名')
    parser.add_argument('--mrna_emb_col', type=str, default='mrna_emb',
                        help='CSV 中 mRNA embedding 列名')
    parser.add_argument('--sirna_col', type=str, default='sirna',
                        help='CSV 中 siRNA 序列列名')
    parser.add_argument('--mrna_col', type=str, default='mrna',
                        help='CSV 中 mRNA 序列列名')

    parser.add_argument('--dp', type=float, default=0.1,
                        help='模型 dropout 概率 (必须与训练时一致)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出文件路径 (CSV 格式), 默认输出到 stdout')
    parser.add_argument('--verbose', '-v', action='store_true', default=True)

    args = parser.parse_args()

    _lazy_imports()
    device = get_device()
    if args.verbose:
        print(f"设备: {device}", file=sys.stderr)

    model = load_model(args.checkpoint, device, dp=args.dp)
    model.eval()

    pssm = None
    if args.pssm:
        pssm = np.load(args.pssm, allow_pickle=True)
        if args.verbose:
            print(f"PSSM 已加载: {args.pssm}", file=sys.stderr)

    rnafm_model = None
    alphabet = None
    need_rnafm = (args.sirna and not (args.sirna_emb and args.mrna_emb)) or \
                 (args.csv and not (args.emb_dir or args.sirna_emb_col or args.mrna_emb_col))

    if need_rnafm:
        if args.verbose:
            print("加载 RNA-FM 模型...", file=sys.stderr)
        rnafm_model, alphabet = load_rnafm_model(device)

    if args.sirna:
        row = {'sirna': args.sirna}
        if args.mrna:
            row['mrna'] = args.mrna
        elif args.mrna_full and args.position is not None:
            row['mrna'] = extract_mrna_window(args.mrna_full, args.position)
        elif args.sirna_emb and args.mrna_emb:
            row['sirna_emb'] = args.sirna_emb
            row['mrna_emb'] = args.mrna_emb
        else:
            parser.error("单条模式需要 --mrna (59-nt 窗口) 或 --mrna_full + --position")

        use_emb_cols = bool(args.sirna_emb and args.mrna_emb)
        preds = run_predictions(
            model, device, [row], pssm=pssm,
            rnafm_model=rnafm_model, alphabet=alphabet,
            sirna_emb_col=args.sirna_emb_col if use_emb_cols else None,
            mrna_emb_col=args.mrna_emb_col if use_emb_cols else None,
            verbose=False,
        )
        result = pd.DataFrame({
            'sirna': [args.sirna],
            'predicted_inhibition': preds,
        })
        if args.mrna:
            result['mrna_window'] = args.mrna
        elif args.mrna_full and args.position is not None:
            result['mrna_window'] = extract_mrna_window(args.mrna_full, args.position)

    elif args.csv:
        df = pd.read_csv(args.csv)
        if args.verbose:
            print(f"加载 CSV: {args.csv} ({len(df)} 条)", file=sys.stderr)

        use_emb_cols = args.sirna_emb_col in df.columns and args.mrna_emb_col in df.columns
        if not use_emb_cols:
            if args.sirna_col not in df.columns or args.mrna_col not in df.columns:
                parser.error(
                    f"CSV 缺少序列列 ('{args.sirna_col}', '{args.mrna_col}') "
                    f"或 embedding 列 ('{args.sirna_emb_col}', '{args.mrna_emb_col}')"
                )

        if pssm is None and not use_emb_cols:
            all_sirnas = df[args.sirna_col].str.upper().str.replace('T', 'U').values
            pssm = create_pssm(all_sirnas)
            if args.verbose:
                print("PSSM 已从输入序列自动计算", file=sys.stderr)

        sequences = df.to_dict('records')
        preds = run_predictions(
            model, device, sequences, pssm=pssm,
            rnafm_model=rnafm_model, alphabet=alphabet,
            emb_dir=args.emb_dir,
            sirna_emb_col=args.sirna_emb_col if use_emb_cols else None,
            mrna_emb_col=args.mrna_emb_col if use_emb_cols else None,
            verbose=args.verbose,
        )
        result = df.copy()
        result['predicted_inhibition'] = preds
    else:
        parser.error("请提供 --sirna 或 --csv")

    if args.output:
        result.to_csv(args.output, index=False)
        if args.verbose:
            print(f"结果已保存: {args.output}", file=sys.stderr)
    else:
        if args.csv:
            result.to_csv(sys.stdout, index=False)
        else:
            for _, r in result.iterrows():
                print(f"siRNA: {r['sirna']}")
                if 'mrna_window' in r:
                    print(f"mRNA窗口: {r['mrna_window']}")
                print(f"预测抑制效率: {r['predicted_inhibition']:.4f}")
                print()


if __name__ == '__main__':
    main()
