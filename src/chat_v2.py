import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import OmnertShakespeare
from v2_config import V2ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(
        ROOT / "checkpoints-v2" / "instruct" / "final.pt",
        map_location=device,
        weights_only=False,
    )
    model = OmnertShakespeare(V2ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    tokenizer = Tokenizer.from_file(
        str(ROOT / "tokenizer" / "v2" / "omnert-126m-tokenizer.json")
    )
    eos = tokenizer.token_to_id("<|endoftext|>")

    print("OMNERT-126M V2 - /quit to exit")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in {"/quit", "/exit"}:
            break
        ids = tokenizer.encode(f"<|user|>\n{prompt}\n<|assistant|>\n").ids
        input_ids = torch.tensor([ids], device=device)
        output = model.generate(
            input_ids,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            eos,
        )[0].tolist()
        text = tokenizer.decode(output[len(ids):], skip_special_tokens=True).strip()
        print("Omnert:", text, "\n")


if __name__ == "__main__":
    main()
