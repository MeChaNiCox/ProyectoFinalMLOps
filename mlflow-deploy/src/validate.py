import json
import argparse
import sys

def main(metrics_path: str, threshold: float):
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    acc = m.get("accuracy", 0.0)
    print(f"[validate] accuracy={acc:.4f}  threshold={threshold:.4f}")
    if acc < threshold:
        print("[validate] ❌ Modelo no cumple el umbral mínimo.")
        sys.exit(1)
    print("[validate] ✅ OK - Modelo válido.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="artifacts/metrics.json")
    p.add_argument("--threshold", type=float, default=0.65)  
    args = p.parse_args()
    main(args.metrics, args.threshold)
