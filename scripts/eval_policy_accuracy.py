import argparse
import json
from pathlib import Path

import numpy as np
import torch

from doa_rl.assets import load_H, load_W
from doa_rl.data import DoADataset, create_dataloader
from doa_rl.model.transformer import TransformerPolicy
from doa_rl.text import Vocab, pad_sequences
from features.tokenizers import PatchTokenizer, NMFTokenizer, direction_projection_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=str, required=True)
    ap.add_argument('--vocab', type=str, required=True)
    ap.add_argument('--tf-path', type=str, required=True)
    ap.add_argument('--w-path', type=str, required=True)
    ap.add_argument('--test-root', type=str, required=True)
    ap.add_argument('--feature', type=str, choices=['patch','nmf'], default='patch')
    ap.add_argument('--add-dir-tokens', action='store_true')
    ap.add_argument('--s-mode', type=str, choices=['S1','S2'], default='S1')
    ap.add_argument('--nmf-iter', type=int, default=50)
    ap.add_argument('--nmf-l1', type=float, default=0.0)
    args = ap.parse_args()

    H, angles = load_H(args.tf_path)
    W = load_W(args.w_path)

    # Data (deterministic order)
    ds = DoADataset(args.test_root, angles.tolist())
    dl = create_dataloader(ds, batch_size=1, shuffle=False)

    # Tokenization
    H_np = H.numpy(); W_np = W.numpy()
    if args.feature == 'nmf':
        fea = NMFTokenizer(W=W_np, n_iter=args.nmf_iter, mode=args.s_mode, l1=args.nmf_l1)
    else:
        fea = PatchTokenizer()

    with open(args.vocab, 'r') as f:
        itos = json.load(f)
    vocab = Vocab([])
    # rebuild exactly same mapping
    for tok in itos:
        vocab.add_token(tok)

    policy = TransformerPolicy(vocab_size=len(vocab.itos), n_dirs=H.shape[1])
    state = torch.load(args.policy, map_location='cpu')
    policy.load_state_dict(state)
    policy.eval()

    correct = 0
    n = 0
    with torch.no_grad():
        for item in dl:
            Y = item['Y'].squeeze(0).numpy()
            if args.feature == 'nmf':
                toks, _ = fea(Y, H=H_np)
            else:
                toks = fea(Y)
            if args.add_dir_tokens:
                toks = toks + direction_projection_tokens(Y.mean(axis=1), H_np)
            ids = vocab.encode(toks, add_cls=True)
            input_ids, attn = pad_sequences([ids], vocab.pad_id)
            logits = policy(input_ids, attn)
            pred = int(torch.argmax(logits, dim=-1).item())
            gt = int(item['angle_index'])
            correct += int(pred == gt)
            n += 1
    acc = 100.0 * correct / max(n, 1)
    print({"n": n, "acc_top1": acc})


if __name__ == '__main__':
    main()
