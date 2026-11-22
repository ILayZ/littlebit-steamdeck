"""Score a saved TinyStories model in a fresh Linux process."""
import argparse
import json
import resource
from pathlib import Path
from experiment import evaluate, reload_model, tokenize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("tokenizer", type=Path)
    args = parser.parse_args()
    texts = Path(__file__).with_name("fixture.txt").read_text().splitlines()
    result = evaluate(reload_model(args.archive), list(tokenize(args.tokenizer, texts)))
    result["peak_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(result, indent=2))
