from typing import List, Dict, Tuple
import torch


class Vocab:
    def __init__(self, specials: List[str] = None):
        self.itos: List[str] = []
        self.stoi: Dict[str, int] = {}
        self.specials = specials or ["[PAD]", "[CLS]", "[SEP]"]
        for sp in self.specials:
            self.add_token(sp)

    def add_token(self, tok: str) -> int:
        if tok not in self.stoi:
            idx = len(self.itos)
            self.stoi[tok] = idx
            self.itos.append(tok)
        return self.stoi[tok]

    def build(self, token_lists: List[List[str]]):
        for toks in token_lists:
            for t in toks:
                self.add_token(t)

    @property
    def pad_id(self) -> int:
        return self.stoi["[PAD]"]

    @property
    def cls_id(self) -> int:
        return self.stoi["[CLS]"]

    def encode(self, toks: List[str], add_cls: bool = True) -> List[int]:
        ids = [self.cls_id] if add_cls else []
        for t in toks:
            ids.append(self.stoi.get(t, self.pad_id))
        return ids


def pad_sequences(seqs: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    L = max(len(s) for s in seqs) if seqs else 0
    input_ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, : len(s)] = 1
    return input_ids, attn

