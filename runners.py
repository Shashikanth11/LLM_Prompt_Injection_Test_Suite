import pandas as pd
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from config import E_COMMERCE_SYSTEM_PROMPT_TEMPLATE
from chat_helpers import chat_ollama
from detectors import detect_refusal, detect_system_leak, detect_canary_leak, benign_success

def run_suite_on_dataset(model: str, dataset_df: pd.DataFrame, temperature: float = 0.2, max_examples=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    canary = f"CANARY_{ts}"
    system = E_COMMERCE_SYSTEM_PROMPT_TEMPLATE.format(canary=canary)

    df = dataset_df.copy()
    if max_examples:
        df = df.head(max_examples)

    records = []
    for i, row in df.iterrows():
        prompt = row["prompt"]
        label = row["label"]
        tag = f"{label}_{i}"

        try:
            out = chat_ollama(model, system, prompt, temperature=temperature)
            text = out["text"]
            rec = {
                "type": label,
                "tag": tag,
                "system_prompt": system,
                "prompt": prompt,
                "response": text,
                "elapsed_ms": out["elapsed_ms"],
                "refusal": detect_refusal(text),
                "attack_success": detect_canary_leak(text, canary) if label=="attack" else None,
                "benign_success": benign_success(text) if label=="benign" else None,
                "canary_leak": detect_canary_leak(text, canary),
                "system_leak_hint": detect_system_leak(text),
                "chars": len(text or "")
            }
        except Exception as e:
            rec = {
                "type": label,
                "tag": tag,
                "system_prompt": system,
                "prompt": prompt,
                "response": f"ERROR: {e}",
                "elapsed_ms": None,
                "refusal": False,
                "attack_success": False,
                "benign_success": False,
                "canary_leak": False,
                "system_leak_hint": False,
                "chars": 0
            }

        records.append(rec)

    result_df = pd.DataFrame.from_records(records)

    # -----------------------------
    # Compute metrics
    # -----------------------------
    benign_df = result_df[result_df['type'] == 'benign']
    attack_df = result_df[result_df['type'] == 'attack']

    metrics = {
        "model": model,
        "n_benign": len(benign_df),
        "n_attacks": len(attack_df),
        "benign_success_rate": float(benign_df["benign_success"].mean()) if len(benign_df) else None,
        "refusal_rate_benign": float(benign_df["refusal"].mean()) if len(benign_df) else None,
        "attack_success_rate": float(attack_df["attack_success"].mean()) if len(attack_df) else None,
        "refusal_rate_attacks": float(attack_df["refusal"].mean()) if len(attack_df) else None,
        "median_latency_ms": int(result_df["elapsed_ms"].dropna().median()) if len(result_df.dropna(subset=['elapsed_ms'])) else None,
        "median_chars": int(result_df["chars"].median()) if len(result_df) else None,
        "timestamp": ts
    }

    # -----------------------------
    # Save CSV + JSON metrics
    # -----------------------------
    csv_path = f"results/ecommerce_suite_{ts}.csv"
    json_path = f"results/ecommerce_suite_{ts}_summary.json"
    result_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # -----------------------------
    # Generate simple plots
    # -----------------------------
    plot_path = f"results/ecommerce_suite_{ts}_plots.png"
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    # Attack success / Benign success
    sns.barplot(x=["Attack Success Rate", "Benign Success Rate"],
                y=[metrics["attack_success_rate"] or 0, metrics["benign_success_rate"] or 0],
                ax=axes[0])
    axes[0].set_ylim(0,1)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Prompt Injection Success Rates")

    # Refusal rates
    sns.barplot(x=["Refusal Attacks", "Refusal Benign"],
                y=[metrics["refusal_rate_attacks"] or 0, metrics["refusal_rate_benign"] or 0],
                ax=axes[1])
    axes[1].set_ylim(0,1)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Refusal Rates")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {"csv": csv_path, "summary": json_path, "plots": plot_path, "metrics": metrics, "df": result_df}
