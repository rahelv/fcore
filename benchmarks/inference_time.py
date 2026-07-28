"""
inference_time.py — latency benchmark (ms per image)
Measures the time for one forward pass. (batch size 1)

Runs one model per invocation.

    python -u inference_time.py --model [vitb32|siglip2|resnet]

ZED Box: pin clocks first so the numbers are stable
    sudo nvpmodel -m 0 && sudo jetson_clocks
"""

import argparse
import numpy as np
import torch

from runners import MODEL_KEYS, build_runner # TODO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_KEYS, required=True) # TODO: 3 models (resnet/other 2)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default=None, help="txt file to save results (default: <model>_latency.txt)")
    args = ap.parse_args()

    # Print to console AND collect the same lines for the txt file.
    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    # Require CUDA, otherwise stop.
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")
    device = torch.device("cuda")
    log(f"Device: {torch.cuda.get_device_name(0)}  |  torch {torch.__version__}  |  model: {args.model}")

    runner = build_runner(args.model, device)
    x = runner.make_input(1, device)  # batch size = 1
    log(f"Runner: {runner.name}  |  input {runner.input_size}px")

    timings = np.zeros(args.reps)

    # GPU warmup (discarded)
    for _ in range(args.warmup):
        runner.forward(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Measure
    for rep in range(args.reps):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        runner.forward(x)
        ender.record()
        torch.cuda.synchronize()
        timings[rep] = starter.elapsed_time(ender)  # ms

    log(f"\n{runner.name} — latency (bs=1, n={args.reps}, {args.warmup} warmup excluded):")
    log(f"  mean   = {timings.mean():7.2f} ms")
    log(f"  median = {np.median(timings):7.2f} ms")
    log(f"  std    = {timings.std():7.2f} ms")
    log(f"  p95    = {np.percentile(timings, 95):7.2f} ms")
    log(f"  min    = {timings.min():7.2f} ms")
    log(f"  max    = {timings.max():7.2f} ms")
    log(f"  peak GPU memory = {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Save the same output to a txt file.
    out = args.out or f"{args.model}_latency.txt"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
