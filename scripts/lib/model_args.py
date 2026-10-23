"""Read the architecture used by the Qwen3-4B and Llama-3.2-3B experiments."""
import json
import sys
from pathlib import Path


def integer_argument(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or (isinstance(value, float) and not value.is_integer()) or value <= 0):
        raise SystemExit("%s must be a positive integer" % name)
    return str(int(value))


def main():
    model_dir, architecture, dry_run = sys.argv[1:]
    path = Path(model_dir) / "config.json"
    if path.is_file():
        config = json.loads(path.read_text())
    elif dry_run == "1":
        defaults = {
            "qwen3": dict(model_type="qwen3", num_hidden_layers=36, hidden_size=2560,
                          intermediate_size=9728, num_attention_heads=32, num_key_value_heads=8,
                          rms_norm_eps=1e-6, rope_theta=5000000, vocab_size=151936,
                          head_dim=128, tie_word_embeddings=True, max_position_embeddings=40960),
            "llama": dict(model_type="llama", num_hidden_layers=28, hidden_size=3072,
                          intermediate_size=8192, num_attention_heads=24, num_key_value_heads=8,
                          rms_norm_eps=1e-5, rope_theta=500000, vocab_size=128256,
                          head_dim=128, tie_word_embeddings=True, max_position_embeddings=131072,
                          rope_scaling={"rope_type": "llama3", "factor": 32.0}),
        }
        config = defaults[architecture]
        print("DRY_RUN: using reference architecture; actual runs read config.json.", file=sys.stderr)
    else:
        raise SystemExit("MODEL_DIR must contain config.json")
    if config.get("model_type") != architecture or architecture not in ("qwen3", "llama"):
        raise SystemExit("This task requires MODEL_ARCH=%s; got model_type=%s" %
                         (architecture, config.get("model_type")))
    if config.get("attention_bias", False) or config.get("mlp_bias", False):
        raise SystemExit("These experiment configurations require bias-free linear layers")
    if config.get("hidden_act", "silu") != "silu":
        raise SystemExit("These experiment configurations require SiLU/SwiGLU")
    args = ["--swiglu", "--group-query-attention", "--use-rotary-position-embeddings",
            "--disable-bias-linear", "--normalization", "RMSNorm"]
    mapping = {
        "--num-layers": "num_hidden_layers", "--hidden-size": "hidden_size",
        "--ffn-hidden-size": "intermediate_size", "--num-attention-heads": "num_attention_heads",
        "--num-query-groups": "num_key_value_heads", "--norm-epsilon": "rms_norm_eps",
        "--rotary-base": "rope_theta", "--vocab-size": "vocab_size",
        "--max-position-embeddings": "max_position_embeddings",
    }
    for flag, key in mapping.items():
        value = str(config[key]) if flag == "--norm-epsilon" else integer_argument(config[key], key)
        args.extend([flag, value])
    head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    args.extend(["--kv-channels", integer_argument(head_dim, "head_dim")])
    if not config.get("tie_word_embeddings", False):
        args.append("--untie-embeddings-and-output-weights")
    if architecture == "qwen3":
        args.append("--qk-layernorm")
        if config.get("rope_scaling"):
            raise SystemExit("Scaled Qwen RoPE is not configured by this experiment launcher")
    elif config.get("rope_scaling"):
        scaling = config["rope_scaling"]
        if scaling.get("rope_type", scaling.get("type")) != "llama3":
            raise SystemExit("Expected Llama 3 RoPE scaling")
        args.extend(["--use-rope-scaling", "--rotary-scaling-factor", str(scaling["factor"])])
    print("\n".join(args))


if __name__ == "__main__":
    main()
