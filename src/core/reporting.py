from datetime import datetime
import os

import numpy as np
import pandas as pd


ABLATION_MODELS = [
    "full_model",
    "w_o_semantics",
    "w_o_reward_shaping",
    "w_o_semantic_prior",
    "w_o_confidence",
    "w_o_partial_offloading",
    "w_o_battery_awareness",
    "w_o_queue_awareness",
    "w_o_mobility_features",
]

REQUIRED_COLUMNS = [
    "run_id",
    "timestamp",
    "config_seed",
    "config_model_type",
    "config_semantic_mode",
    "config_total_tasks",
    "metric_success_rate",
    "metric_avg_reward",
    "metric_p95_latency",
    "metric_avg_energy",
    "metric_qoe",
    "config_batch_id",
    "config_eval_group",
]


def _load_experiment_df(csv_path):
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.read_csv(csv_path)
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _latest_batch(df, batch_prefix):
    if df.empty or "config_batch_id" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    batch_df = df[df["config_batch_id"].astype(str).str.startswith(batch_prefix)].copy()
    if batch_df.empty:
        return batch_df

    batch_df = batch_df.sort_values("timestamp")
    latest_batch_id = batch_df["config_batch_id"].iloc[-1]
    return batch_df[batch_df["config_batch_id"] == latest_batch_id].copy()


def _latest_batch_for_group(df, eval_group, batch_prefix=None):
    if df.empty:
        return pd.DataFrame(columns=df.columns)

    batch_df = df[df["config_eval_group"].astype(str) == str(eval_group)].copy()
    if batch_df.empty:
        return batch_df

    if batch_prefix:
        batch_df = batch_df[batch_df["config_batch_id"].astype(str).str.startswith(batch_prefix)].copy()
        if batch_df.empty:
            return batch_df

    batch_df = batch_df.sort_values("timestamp")
    latest_batch_id = batch_df["config_batch_id"].iloc[-1]
    return batch_df[batch_df["config_batch_id"] == latest_batch_id].copy()


def _latest_batch_for_groups(df, eval_groups, batch_prefixes=None):
    if df.empty:
        return pd.DataFrame(columns=df.columns)

    eval_groups = [str(group) for group in eval_groups]
    batch_df = df[df["config_eval_group"].astype(str).isin(eval_groups)].copy()
    if batch_df.empty:
        return batch_df

    if batch_prefixes:
        mask = False
        for prefix in batch_prefixes:
            mask = mask | batch_df["config_batch_id"].astype(str).str.startswith(prefix)
        batch_df = batch_df[mask].copy()
        if batch_df.empty:
            return batch_df

    batch_df = batch_df.sort_values("timestamp")
    latest_batch_id = batch_df["config_batch_id"].iloc[-1]
    return batch_df[batch_df["config_batch_id"] == latest_batch_id].copy()


def _aggregate_results(df):
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby("config_model_type")[
            ["metric_success_rate", "metric_avg_reward", "metric_p95_latency", "metric_avg_energy", "metric_qoe"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.values]
    return grouped.sort_values("metric_success_rate_mean", ascending=False)


def _safe_std(value):
    return 0.0 if pd.isna(value) else float(value)


def _percent_mean_std(mean_value, std_value):
    return f"{float(mean_value) * 100:.2f}% +- {_safe_std(std_value) * 100:.2f}"


def _scalar_mean_std(mean_value, std_value, precision):
    return f"{float(mean_value):.{precision}f} +- {_safe_std(std_value):.{precision}f}"


def _display_eval_group(eval_group):
    mapping = {
        "phase5_retraining_multiseed": "phase5_baseline_retraining",
        "ablation_retraining_multiseed": "phase5_ablation_retraining",
        "ablation_multiseed_retrained_ppo": "phase5_ablation_evaluation",
    }
    return mapping.get(str(eval_group), str(eval_group))


def _build_batch_overview(df):
    if df.empty:
        return "Henüz batch verisi yok.\n"

    batch_df = (
        df[df["config_batch_id"].astype(str).str.len() > 0]
        .groupby(["config_batch_id", "config_eval_group"], dropna=False)
        .agg(
            last_update=("timestamp", "max"),
            runs=("run_id", "count"),
            models=("config_model_type", "nunique"),
            tasks=("config_total_tasks", "sum"),
        )
        .reset_index()
        .sort_values("last_update", ascending=False)
        .head(10)
    )

    if batch_df.empty:
        return "Henüz batch verisi yok.\n"

    lines = [
        "| Batch ID | Eval Group | Last Update | Runs | Models | Total Tasks |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in batch_df.iterrows():
        last_update = row["last_update"].isoformat() if pd.notna(row["last_update"]) else "-"
        lines.append(
            f"| {row['config_batch_id']} | {_display_eval_group(row['config_eval_group'] or '-')} | "
            f"{last_update} | {int(row['runs'])} | {int(row['models'])} | {int(row['tasks'])} |"
        )
    return "\n".join(lines) + "\n"


def _build_reading_guide():
    lines = [
        "## Bu Rapor Nasil Okunmali",
        "",
        "- `Success Rate`: deadline icinde tamamlanan task oranidir. Yuksek olmasi iyidir.",
        "- `P95 Latency`: en yavas kuyrugun davranisini gosterir. Ortalama degil, tail-latency odaklidir. Dusuk olmasi iyidir.",
        "- `Avg Energy`: task basina ortalama enerji tuketimidir. Dusuk olmasi iyidir.",
        "- `QoE`: success ve latency'nin birlesik, daha yorumlayici bir ozetidir.",
        "- `Delta vs Full`: ilgili ablation varyantinin Full Model'e gore success farkidir.",
        "",
        "Bu raporda iki farkli deney tipi birlikte bulunur:",
        "- `evaluation`: mevcut checkpoint ailesi farkli seed'lerde test edilir.",
        "- `retraining`: model her seed icin sifirdan yeniden egitilir.",
        "",
        "Faz 5 yorumu yaparken retraining bolumleri, evaluation-only bolumlerinden daha guclu kanit olarak okunmalidir.",
    ]
    return "\n".join(lines) + "\n"


def _build_phase_boundary_note():
    lines = [
        "## Faz Siniri",
        "",
        "Bu rapordaki baseline ve ablation sonuclari Faz 5 kapsaminda degerlendirilmelidir.",
        "Cunku burada cevaplanan soru, mevcut model ailesi ve semantic bilesenlerin katkilarinin ne oldugudur.",
        "",
        "Faz 5 kapsaminda kalan isler:",
        "- baseline karsilastirmalarini daha saglam hale getirmek",
        "- ablation sonuclarini coklu seed ile daha savunulabilir yapmak",
        "- gerekiyorsa ayni sentetik/simule ortamda multi-seed retraining eklemek",
        "",
        "Faz 6 ancak trace-driven egitim ve trace-driven evaluation ana akisa gectigimizde baslar.",
        "Yani gercek gecis noktasi, sentetik episode yerine trace tabanli is yukleriyle modeli yeniden egitmek ve bu sonuclari raporlamaktir.",
    ]
    return "\n".join(lines) + "\n"


def _build_multiseed_explanation():
    lines = [
        "## Neden Multi-Seed Retraining",
        "",
        "`Multi-seed evaluation` ile `multi-seed retraining` ayni sey degildir.",
        "",
        "- `Multi-seed evaluation`: ayni egitilmis model farkli evaluation seed'lerinde test edilir.",
        "- `Multi-seed retraining`: model her seed icin sifirdan yeniden egitilir ve sonra karsilastirilir.",
        "",
        "RL ajanlari random initialization, experience ordering, environment stochasticity ve exploration farklari nedeniyle seed'e hassastir.",
        "Bu yuzden tek bir seed'de iyi gorunen model baska bir seed'de ayni sekilde davranmayabilir.",
        "",
        "Bu islemi yapmamizin temel nedenleri sunlardir:",
        "- tek bir sansli training kosusuna asiri guvenmemek",
        "- algoritmalarin gercekten daha iyi olup olmadigini varyansla birlikte okumak",
        "- Faz 5 bulgularini Faz 6'ya tasimadan once daha savunulabilir hale getirmek",
        "- sonraki trace-driven asamaya daha saglam bir sentetik temel ile gecmek",
        "",
        "Kisaca: multi-seed evaluation, mevcut modelin test-dayanikliligini; multi-seed retraining ise egitim surecinin kendisinin ne kadar kararlı oldugunu gosterir.",
    ]
    return "\n".join(lines) + "\n"


def _build_methodology_notes():
    lines = [
        "## Metodoloji Notlari",
        "",
        "- Evaluation-only sonuclar evaluation-seed cesitliligi saglar, fakat training-seed cesitliligi saglamaz.",
        "- Retraining bolumleri ise training-seed cesitliligi ekler; Faz 5 kapanis yorumu icin asil dayanak bunlar olmalidir.",
        "- Bazi varyantlarin birbirine cok yakin cikmasi, ilgili bilesenin etkisiz oldugunu degil; mevcut state, reward veya env tasariminin bu farki yeterince ayristiramadigini da gosterebilir.",
        "- Ozellikle `w_o_reward_shaping` ve `w_o_queue_awareness` sonuclarini bu gozle okumak gerekir.",
        "- `configs/ablation.yaml` tek kanonik ablation config dosyasidir; `mode: evaluation` ve `mode: retrain` ayni dosyadan yonetilir.",
    ]
    return "\n".join(lines) + "\n"


def _build_workflow_map():
    lines = [
        "## Kanonik Deney Akisi",
        "",
        "Bu repo icinde Faz 5 icin sade akisin hangi dosyalardan gectigi burada ozetlenir.",
        "",
        "- Ortak egitim recetesi: `configs/synthetic_rl_training.yaml`",
        "- Baseline retraining orkestrasyonu: `configs/phase5_baseline_retraining.yaml`",
        "- Ablation config ve mod secimi: `configs/ablation.yaml`",
        "- Baseline retraining scripti: `experiments/run_baseline_retraining.py`",
        "- Ablation scripti: `experiments/run_ablation_study.py`",
        "- Kanonik rapor: `results/tables/offloading_experiment_report.md`",
        "",
        "Model ciktilari agent bazli klasorlerde tutulur:",
        "- PPO baseline retraining: `models/ppo/phase5_baseline_retraining/`",
        "- DQN baseline retraining: `models/dqn/phase5_baseline_retraining/`",
        "- A2C baseline retraining: `models/a2c/phase5_baseline_retraining/`",
        "- PPO ablation retraining varyantlari: `models/ppo/phase5_ablation_retraining/<varyant>/`",
    ]
    return "\n".join(lines) + "\n"


def _build_retraining_section(df):
    if df.empty:
        return (
            "## Faz 5 Baseline Retraining\n\n"
            "Bu bolum icin henuz baseline retraining verisi yok. "
            "Gercek multi-seed retraining calistirildiginda burada farkli train seed'lerle egitilen modellerin aggregate sonuclari yer alir.\n"
        )

    grouped = _aggregate_results(df)
    lines = [
        "## Faz 5 Baseline Retraining",
        "",
        "Bu bolum, ayni modellerin sadece farkli evaluation seed'lerde test edilmesini degil, farkli train seed'lerle sifirdan yeniden egitilmesini ozetler.",
        "Bu nedenle metodolojik olarak baseline multi-seed evaluation bolumunden daha gucludur.",
        "",
        "| Model | Success Rate (mean +- std) | Avg Reward (mean +- std) | P95 Latency (mean +- std) | Avg Energy (mean +- std) | QoE (mean +- std) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in grouped.iterrows():
        lines.append(
            f"| {row['config_model_type']} | "
            f"{_percent_mean_std(row['metric_success_rate_mean'], row['metric_success_rate_std'])} | "
            f"{_scalar_mean_std(row['metric_avg_reward_mean'], row['metric_avg_reward_std'], 2)} | "
            f"{_scalar_mean_std(row['metric_p95_latency_mean'], row['metric_p95_latency_std'], 3)} | "
            f"{_scalar_mean_std(row['metric_avg_energy_mean'], row['metric_avg_energy_std'], 4)} | "
            f"{_scalar_mean_std(row['metric_qoe_mean'], row['metric_qoe_std'], 2)} |"
        )

    lines.extend(
        [
            "",
            "Bu bolum Faz 5 kapanisi icin kritik kabul edilmelidir; cunku seed'e bagli sans etkisini azaltir ve model karsilastirmasini daha savunulabilir hale getirir.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_baseline_section(df):
    if df.empty:
        return "## Baseline Multi-Seed Sonuclari\n\nBu bolum icin henuz veri yok.\n"

    grouped = _aggregate_results(df)
    lines = [
        "## Baseline Multi-Seed Sonuclari",
        "",
        "Bu tablo ayni egitilmis modellerin farkli evaluation seed'lerinde nasil davrandigini ozetler.",
        "Not: Bu bolum multi-seed evaluation'dir; multi-seed retraining degildir.",
        "",
        "| Model | Success Rate (mean +- std) | Avg Reward (mean +- std) | P95 Latency (mean +- std) | Avg Energy (mean +- std) | QoE (mean +- std) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in grouped.iterrows():
        lines.append(
            f"| {row['config_model_type']} | "
            f"{_percent_mean_std(row['metric_success_rate_mean'], row['metric_success_rate_std'])} | "
            f"{_scalar_mean_std(row['metric_avg_reward_mean'], row['metric_avg_reward_std'], 2)} | "
            f"{_scalar_mean_std(row['metric_p95_latency_mean'], row['metric_p95_latency_std'], 3)} | "
            f"{_scalar_mean_std(row['metric_avg_energy_mean'], row['metric_avg_energy_std'], 4)} | "
            f"{_scalar_mean_std(row['metric_qoe_mean'], row['metric_qoe_std'], 2)} |"
        )
    return "\n".join(lines) + "\n"


def _build_ablation_section(df, figure_path):
    if df.empty:
        return "## Ablation Multi-Seed Sonuclari\n\nBu bolum icin henuz veri yok.\n"

    grouped = _aggregate_results(df)
    baseline_row = grouped[grouped["config_model_type"] == "full_model"]
    baseline_success = float(baseline_row["metric_success_rate_mean"].iloc[0]) if not baseline_row.empty else 0.0

    lines = [
        "## Ablation Multi-Seed Sonuclari",
        "",
        "Bu tablo semantic bilesenlerin bireysel etkisini coklu evaluation seed uzerinden gosterir.",
        "Full Model: semantics, reward shaping, semantic prior, confidence weighting, partial offloading, battery awareness, queue awareness ve mobility features acik olan temel sistemdir.",
        "",
        "| Ablation Model | Success Rate (mean +- std) | Avg Reward (mean +- std) | P95 Latency (mean +- std) | Avg Energy (mean +- std) | QoE (mean +- std) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in grouped.iterrows():
        lines.append(
            f"| {row['config_model_type']} | "
            f"{_percent_mean_std(row['metric_success_rate_mean'], row['metric_success_rate_std'])} | "
            f"{_scalar_mean_std(row['metric_avg_reward_mean'], row['metric_avg_reward_std'], 2)} | "
            f"{_scalar_mean_std(row['metric_p95_latency_mean'], row['metric_p95_latency_std'], 3)} | "
            f"{_scalar_mean_std(row['metric_avg_energy_mean'], row['metric_avg_energy_std'], 4)} | "
            f"{_scalar_mean_std(row['metric_qoe_mean'], row['metric_qoe_std'], 2)} |"
        )

    lines.extend(
        [
            "",
            "### Delta Analizi",
            "",
            "Delta analizi, her ablation senaryosunun Full Model'e gore ne kadar iyilestigini veya kotulestigini gosterir.",
            "Pozitif delta, ilgili varyantin Full Model'den daha yuksek success verdigini; negatif delta ise daha kotu oldugunu anlatir.",
            "Contribution kolonu, cikarilan bilesenin yaklasik etkisini `-delta` olarak okumayi kolaylastirir.",
            "",
            f"Baseline (Full Model): {baseline_success * 100:.2f}%",
            "",
            "| Ablation | Mean Success % | Delta vs Full | Contribution |",
            "|---|---:|---:|---:|",
        ]
    )

    for _, row in grouped.iterrows():
        success = float(row["metric_success_rate_mean"])
        delta = (success - baseline_success) * 100.0
        contribution = 0.0 if row["config_model_type"] == "full_model" else -delta
        lines.append(
            f"| {row['config_model_type']} | {success * 100:.2f}% | {delta:+.2f}% | {contribution:.2f}% |"
        )

    if os.path.exists(figure_path):
        lines.extend(
            [
                "",
                "### Figure",
                "",
                "![Ablation Impact](../figures/ablation_impact.png)",
            ]
        )

    return "\n".join(lines) + "\n"


def _build_ablation_extended_section(df):
    if df.empty:
        return "## Kapsamli Ablation Analizi\n\nBu bolum icin henuz veri yok.\n"

    grouped = _aggregate_results(df)
    baseline_row = grouped[grouped["config_model_type"] == "full_model"]
    baseline_success = float(baseline_row["metric_success_rate_mean"].iloc[0]) if not baseline_row.empty else 0.0

    lines = [
        "## Kapsamli Ablation Analizi",
        "",
        "Bu bolum, ablation sonuclarinin yonetici ozeti olarak tek bakista okunmasi icin hazirlandi.",
        "Amac, ablation sonuclarini success, enerji, tail-latency ve QoE eksenlerinde hizli karsilastirmaktir.",
        "",
        "| Ablation Model | Success Rate (mean +- std) | Avg Energy (J) | P95 Latency (s) | QoE Score | Delta vs Baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in grouped.iterrows():
        success_mean = float(row["metric_success_rate_mean"])
        success_std = row["metric_success_rate_std"]
        delta = (success_mean - baseline_success) * 100.0
        delta_text = "0.00% (Baseline)" if row["config_model_type"] == "full_model" else f"{delta:+.2f}%"
        lines.append(
            f"| {row['config_model_type']} | "
            f"{_percent_mean_std(success_mean, success_std)} | "
            f"{float(row['metric_avg_energy_mean']):.3f} | "
            f"{float(row['metric_p95_latency_mean']):.3f} | "
            f"{float(row['metric_qoe_mean']):.2f} | "
            f"{delta_text} |"
        )

    lines.extend(
        [
            "",
            "### Kisa Yorum",
            "",
            "- `w_o_partial_offloading` success'i cok sert dusurmese de `p95 latency`yi belirgin bicimde kotulestiriyor; partial offloading katkisi daha cok tail-latency tarafinda gorunuyor.",
            "- `w_o_mobility_features` en buyuk negatif etkiyi veriyor; bu da mobilite/distance bilgisinin karar kalitesi icin kritik oldugunu gosteriyor.",
            "- `w_o_battery_awareness` varyantinin Full Model'den bir miktar iyi gorunmesi, mevcut reward tasariminda enerji disiplini ile success optimizasyonu arasinda gerilim olduguna isaret ediyor.",
            "- `w_o_reward_shaping` ve `w_o_queue_awareness` sonuclarinin Full Model'e cok yakin olmasi, bu bilesenlerin etkisinin mevcut protokolde yeterince ayrisamamis olabilecegini dusunduruyor.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_ablation_retraining_section(df):
    if df.empty:
        return (
            "## Faz 5 Ablation Retraining\n\n"
            "Bu bolum icin henuz gercek ablation retraining verisi yok. "
            "Her varyant sifirdan egitildiginde semantic ve fiziksel bilesenlerin gercek katkisi burada gorunur.\n"
        )

    grouped = _aggregate_results(df)
    baseline_row = grouped[grouped["config_model_type"] == "full_model"]
    baseline_success = float(baseline_row["metric_success_rate_mean"].iloc[0]) if not baseline_row.empty else 0.0

    lines = [
        "## Faz 5 Ablation Retraining",
        "",
        "Bu bolum, ablation varyantlarinin sadece test edilmesini degil, her birinin ayri ayri sifirdan yeniden egitilmesini ozetler.",
        "Bu nedenle semantic bilesen katkisini okumak icin en guvenilir Faz 5 tablosu budur.",
        "",
        "| Ablation Model | Success Rate (mean +- std) | Avg Reward (mean +- std) | P95 Latency (mean +- std) | Avg Energy (mean +- std) | QoE (mean +- std) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in grouped.iterrows():
        lines.append(
            f"| {row['config_model_type']} | "
            f"{_percent_mean_std(row['metric_success_rate_mean'], row['metric_success_rate_std'])} | "
            f"{_scalar_mean_std(row['metric_avg_reward_mean'], row['metric_avg_reward_std'], 2)} | "
            f"{_scalar_mean_std(row['metric_p95_latency_mean'], row['metric_p95_latency_std'], 3)} | "
            f"{_scalar_mean_std(row['metric_avg_energy_mean'], row['metric_avg_energy_std'], 4)} | "
            f"{_scalar_mean_std(row['metric_qoe_mean'], row['metric_qoe_std'], 2)} |"
        )

    lines.extend(
        [
            "",
            f"Baseline (Full Model): {baseline_success * 100:.2f}%",
            "",
            "| Ablation | Mean Success % | Delta vs Full | Contribution |",
            "|---|---:|---:|---:|",
        ]
    )

    for _, row in grouped.iterrows():
        success = float(row["metric_success_rate_mean"])
        delta = (success - baseline_success) * 100.0
        contribution = 0.0 if row["config_model_type"] == "full_model" else -delta
        lines.append(
            f"| {row['config_model_type']} | {success * 100:.2f}% | {delta:+.2f}% | {contribution:.2f}% |"
        )

    lines.extend(
        [
            "",
            "Bu tablo, inference-only ablation sonucundan daha onemlidir; cunku policy'nin hangi sinyallerle yeniden sekillendigini gosterir.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_experiment_report(
    csv_path="results/raw/master_experiments.csv",
    output_path="results/tables/offloading_experiment_report.md",
    figure_path="results/figures/ablation_impact.png",
):
    df = _load_experiment_df(csv_path)
    baseline_df = _latest_batch(df, "baseline_")
    ablation_df = _latest_batch(df[df["config_model_type"].isin(ABLATION_MODELS)].copy(), "ablation_")
    retraining_df = _latest_batch_for_groups(
        df,
        ["phase5_retraining_multiseed", "phase5_baseline_retraining"],
        ["retrain_", "baseline_retrain_"],
    )
    ablation_retraining_df = _latest_batch_for_groups(
        df[df["config_model_type"].isin(ABLATION_MODELS)].copy(),
        ["ablation_retraining_multiseed", "phase5_ablation_retraining"],
        ["ablation_retrain_"],
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("# Task Offloading Experiment Report\n\n")
        handle.write(
            "Bu dosya `results/tables` altindaki tek kanonik okuma noktasi olarak uretilir. "
            "Ham veri kaynagi degismeden `results/raw/master_experiments.csv` icinde tutulur.\n\n"
        )
        handle.write("## Proje Akisi\n\n")
        handle.write(
            "- `models/`: egitilmis ajanlar\n"
            "- `experiments/`: deneyleri kosan script'ler\n"
            "- `results/raw/`: kaynaga en yakin deney loglari\n"
            "- `results/tables/offloading_experiment_report.md`: insanlar icin tek ozet rapor\n"
            "- `results/figures/`: gorseller\n\n"
        )
        handle.write("## Son Batch Ozeti\n\n")
        handle.write(_build_batch_overview(df))
        handle.write("\n")
        handle.write(_build_reading_guide())
        handle.write("\n")
        handle.write(_build_phase_boundary_note())
        handle.write("\n")
        handle.write(_build_multiseed_explanation())
        handle.write("\n")
        handle.write(_build_methodology_notes())
        handle.write("\n")
        handle.write(_build_workflow_map())
        handle.write("\n")
        handle.write(_build_retraining_section(retraining_df))
        handle.write("\n")
        handle.write(_build_ablation_retraining_section(ablation_retraining_df))
        handle.write("\n")
        handle.write(_build_baseline_section(baseline_df))
        handle.write("\n")
        handle.write(_build_ablation_section(ablation_df, figure_path))
        handle.write("\n")
        handle.write(_build_ablation_extended_section(ablation_df))
        handle.write("\n---\n")
        handle.write(f"*Updated: {datetime.now().isoformat()}*\n")
