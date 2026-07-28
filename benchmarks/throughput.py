"""
throughput.py — throughput benchmark (images per second)
=================================================================
Measures how many images the GPU sustains when kept fully busy with a LARGE
batch, reflects the model's true relative compute cost.

Sweeps several batch sizes and reports the best (plateau) throughput.

    python -u throughput.py --model [vitb32|siglip2|resnet]  --batch-sizes 1 8 16 32 64

ZED Box: pin clocks first
    sudo nvpmodel -m 0 && sudo jetson_clocks
"""

import argparse
import torch

from runners import MODEL_KEYS, build_runner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_KEYS, required=True)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default=None, help="txt file to save results (default: <model>_throughput.txt)")
    args = ap.parse_args()

    # Print to console AND collect the same lines for the txt file.
    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")
    device = torch.device("cuda")
    log(f"Device: {torch.cuda.get_device_name(0)}  |  torch {torch.__version__}  |  model: {args.model}")

    runner = build_runner(args.model, device)
    log(f"Runner: {runner.name}  |  input {runner.input_size}px")

    log(f"\n{'batch':>6} {'ms/batch':>10} {'ms/img':>9} {'img/s':>10}")
    best = (0.0, None)
    for bs in args.batch_sizes:
        x = runner.make_input(bs, device) # batch

        # warmup for batch size
        for _ in range(args.warmup):
            runner.forward(x)
        torch.cuda.synchronize()

        # Time reps forward passes of the whole batch, accumulate total seconds
        total_s = 0.0
        for _ in range(args.reps):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            runner.forward(x)
            ender.record()
            torch.cuda.synchronize()
            total_s += starter.elapsed_time(ender) / 1000.0   # ms -> s

        throughput = (args.reps * bs) / total_s # images / second
        ms_batch = total_s / args.reps * 1000
        log(f"{bs:>6} {ms_batch:>10.2f} {ms_batch/bs:>9.2f} {throughput:>10.2f}")
        if throughput > best[0]:
            best = (throughput, bs)

    log(f"\nBest throughput: {best[0]:.2f} img/s at batch size {best[1]}")

    # Save the same output to a txt file.
    out = args.out or f"{args.model}_throughput.txt"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
