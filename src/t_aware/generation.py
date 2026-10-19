"""Temperature-controlled generation with per-temperature resume keys."""
import json
from pathlib import Path


TEMPERATURES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5]


def generate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .engine import choose_device, seed_everything
    from .io import environment, fingerprint, write_json
    source, output = Path(args.input).resolve(), Path(args.output).resolve()
    if source == output:
        raise ValueError("Input and output must differ")
    if output.exists() and not args.resume:
        raise FileExistsError(f"{output}; --resume is required to append")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    prompts, ids = [], set()
    with source.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("id", "prompt", "human_answer")):
                raise ValueError(f"Input line {number} needs nonempty id, prompt, human_answer")
            if row["id"] in ids:
                raise ValueError(f"Repeated source id at line {number}")
            ids.add(row["id"])
            prompts.append(row)
            if args.limit and len(prompts) >= args.limit:
                break
    if not prompts:
        raise ValueError("No prompts")
    done = set()
    if output.exists():
        with output.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = (row["base_id"], float(row["generation_T"]))
                expected_alias = "Human" if key[1] < 0 else args.model_alias
                if row.get("model") != expected_alias:
                    raise ValueError("Existing output contains another generator")
                if key in done:
                    raise ValueError("Existing output has duplicate source/temperature keys")
                if not row.get("generated_text", "").strip():
                    raise ValueError("Existing output has an empty generation; repair a COPY before resuming")
                done.add(key)
    device = choose_device(args.device)
    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype)).to(device).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = output.with_suffix(output.suffix + ".run.json")
    if not args.resume:
        write_json(meta, {"input": fingerprint(source), "model": args.model, "model_alias": args.model_alias,
                         "temperatures": TEMPERATURES, "top_k": args.top_k, "top_p": args.top_p,
                         "seed": args.seed, "max_new_tokens": args.max_new_tokens, "chat_template": args.chat_template,
                         "revision": getattr(model.config, "_commit_hash", None), "environment": environment()})
    else:
        if not meta.is_file():
            raise ValueError("Resume requires the original .run.json generation metadata")
        recorded = json.loads(meta.read_text(encoding="utf-8"))
        for key, value in (("model", args.model), ("model_alias", args.model_alias), ("top_k", args.top_k),
                           ("top_p", args.top_p), ("seed", args.seed), ("max_new_tokens", args.max_new_tokens),
                           ("chat_template", args.chat_template)):
            if recorded.get(key) != value:
                raise ValueError(f"Resume configuration differs: {key}")
        if recorded["input"]["sha256"] != fingerprint(source)["sha256"]:
            raise ValueError("Resume source file changed")
    with output.open("a" if args.resume else "x", encoding="utf-8") as stream:
        for row in prompts:
            for temperature in TEMPERATURES + [-1.0]:
                if (row["id"], temperature) in done:
                    continue
                if temperature < 0:
                    text = row["human_answer"].strip()
                else:
                    # Stable per-source/temperature seed makes resumed items independent of earlier work.
                    import hashlib
                    seed = int(hashlib.sha256(f"{args.seed}:{row['id']}:{temperature}".encode()).hexdigest()[:8], 16)
                    seed_everything(seed)
                    prompt = row["prompt"]
                    if args.chat_template:
                        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                    encoded = tokenizer(prompt, return_tensors="pt")
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    with torch.no_grad():
                        generated = model.generate(**encoded, temperature=temperature, do_sample=True,
                            top_k=args.top_k, top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                            pad_token_id=tokenizer.eos_token_id)
                    text = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    if not text:
                        raise ValueError(f"Empty generation at source {row['id']}, T={temperature}; output can be resumed")
                record = {"id": f"{row['id']}_Human_Anchor" if temperature < 0 else f"{row['id']}_{args.model_alias}_T{temperature}",
                          "base_id": row["id"], "prompt": row["prompt"], "model": "Human" if temperature < 0 else args.model_alias,
                          "generation_T": temperature, "generated_text": text}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                done.add((row["id"], temperature))


def register_generation(commands):
    command = commands.add_parser("generate", help="Generate 13 temperature variants plus a human anchor")
    command.add_argument("--input", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--model", required=True)
    command.add_argument("--model-alias", required=True)
    command.add_argument("--resume", action="store_true")
    command.add_argument("--limit", type=int)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--device", default="auto")
    command.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    command.add_argument("--max-new-tokens", type=int, default=1024)
    command.add_argument("--top-k", type=int, default=50)
    command.add_argument("--top-p", type=float, default=1.0)
    command.add_argument("--chat-template", action="store_true")
    command.set_defaults(handler=generate)
