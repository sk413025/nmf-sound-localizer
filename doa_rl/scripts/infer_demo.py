import argparse
from doa_rl.env.dataset import build_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--prompt", type=str, required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    out = mdl.generate(ids, max_new_tokens=4)
    print(tok.decode(out[0]))


if __name__ == "__main__":
    main()

