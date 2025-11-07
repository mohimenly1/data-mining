from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import re
import textwrap

import arabic_reshaper
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bidi.algorithm import get_display
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class AprioriConfig:
    min_support: float = 0.05
    min_confidence: float = 0.6
    max_rules: int = 30


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicates to avoid overweighting repeated registrations
    df = df.drop_duplicates().copy()

    # Standardise column names (strip whitespace)
    df.columns = [col.strip() for col in df.columns]

    # Handle missing academic year
    df["Academic year"] = (
        df["Academic year"].fillna("Unknown").astype(str).replace({"nan": "Unknown"})
    )

    # Ensure numeric columns are properly typed
    for col in ["date of birth", "Year of registration", "High school success rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive engineered categorical features for association analysis
    df["birth_bucket"] = pd.cut(
        df["date of birth"],
        bins=[df["date of birth"].min() - 1, 1990, 1993, 1996, df["date of birth"].max() + 1],
        labels=["<=1989", "1990-1992", "1993-1995", ">=1996"],
    )

    df["registration_bucket"] = pd.cut(
        df["Year of registration"],
        bins=[
            df["Year of registration"].min() - 1,
            2010,
            2012,
            2014,
            df["Year of registration"].max() + 1,
        ],
        labels=["<=2010", "2011-2012", "2013-2014", ">=2015"],
    )

    df["success_rate_band"] = pd.cut(
        df["High school success rate"],
        bins=[0, 70, 80, 90, 100],
        labels=["<70", "70-79", "80-89", "90-100"],
        include_lowest=True,
    )

    df["Pass or fail?"] = df["Pass or fail?"].str.strip().replace({"Failed": "Failed", "successful": "Successful"})

    return df


def _normalise_value(value: str) -> str:
    value = value.strip()
    value = value.replace(" ", "_")
    return value


def encode_for_apriori(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    temp = df[list(features)].astype(str)
    for col in temp.columns:
        temp[col] = col.replace(" ", "_") + "=" + temp[col].apply(_normalise_value)
    encoded = pd.get_dummies(temp)
    encoded.columns = [col.split("_", 1)[1] if "_" in col else col for col in encoded.columns]
    encoded = encoded.astype(bool)
    return encoded


def run_apriori(
    encoded_df: pd.DataFrame,
    config: AprioriConfig,
    extra_filter: Optional[dict] = None,
) -> pd.DataFrame:
    frequent = apriori(
        encoded_df,
        min_support=config.min_support,
        use_colnames=True,
        verbose=0,
    )

    if frequent.empty:
        return pd.DataFrame()

    rules = association_rules(
        frequent, metric="confidence", min_threshold=config.min_confidence
    )

    if rules.empty:
        return pd.DataFrame()

    if extra_filter:
        for key, value in extra_filter.items():
            rules = rules[rules[key] >= value]

    transform = lambda s: ["{}".format(item) for item in sorted(s)]
    rules["antecedents"] = rules["antecedents"].apply(transform).str.join(", ")
    rules["consequents"] = rules["consequents"].apply(transform).str.join(", ")
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x.split(", ")) if x else 0)

    keep_cols = [
        "antecedents",
        "consequents",
        "antecedent_len",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
    ]

    rules = rules[keep_cols].sort_values(by=["lift", "confidence", "support"], ascending=False)
    return rules.head(config.max_rules).reset_index(drop=True)


def prepare_features_for_clustering(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "date of birth",
        "Year of registration",
        "High school success rate",
        "Gender",
        "city",
        "Secondary system",
        "School system",
        "Specialization ID",
        "Pass or fail?",
    ]

    feature_df = df[feature_cols].copy()
    feature_df["Pass or fail?"] = feature_df["Pass or fail?"].fillna("Unknown")
    categorical_cols = feature_df.select_dtypes(include="object").columns
    feature_df[categorical_cols] = feature_df[categorical_cols].apply(lambda col: col.fillna("Unknown"))

    encoded = pd.get_dummies(feature_df, drop_first=False)
    return encoded


def choose_best_k(X: np.ndarray, random_state: int = 42) -> dict:
    k_values = list(range(2, 9))
    scores = {}

    if X.shape[0] > 50000:
        rng = np.random.default_rng(random_state)
        subset_idx = rng.choice(X.shape[0], 50000, replace=False)
        X_eval = X[subset_idx]
    else:
        X_eval = X

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X_eval)
        score = silhouette_score(X_eval, labels)
        scores[k] = float(score)

    best_k = max(scores, key=scores.get)
    return {"best_k": int(best_k), "scores": scores}


def perform_clustering(encoded_features: pd.DataFrame, random_state: int = 42) -> tuple[np.ndarray, dict]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(encoded_features)

    clustering_info = choose_best_k(scaled, random_state=random_state)
    best_k = clustering_info["best_k"]

    model = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels = model.fit_predict(scaled)

    return labels, clustering_info


def ensure_output_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def summarise_clusters(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    summary = (
        df.groupby(cluster_col)
        .agg(
            students=("Registration number", "count"),
            success_rate_mean=("High school success rate", "mean"),
            pass_rate=("Pass or fail?", lambda x: (x == "Successful").mean()),
        )
        .reset_index()
    )
    summary["success_rate_mean"] = summary["success_rate_mean"].round(2)
    summary["pass_rate"] = (summary["pass_rate"] * 100).round(2)
    return summary


def save_rules(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "antecedents",
                "consequents",
                "antecedent_len",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction",
            ]
        )
    df.to_csv(path, index=False)


_ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF]")


def _contains_arabic(text: Optional[str]) -> bool:
    return bool(text) and bool(_ARABIC_PATTERN.search(text))


def _shape_arabic(text: str) -> str:
    if not _contains_arabic(text):
        return text
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


def _bidi_text(text: str) -> str:
    if not text:
        return text
    shaped = _shape_arabic(text)
    return f"\u200F{shaped}" if _contains_arabic(shaped) else shaped


def _bilingual(en_text: str, ar_text: str) -> str:
    return f"{en_text}\n{_bidi_text(ar_text)}"


def plot_pass_fail_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    counts = (
        df["Pass or fail?"]
        .value_counts()
        .reindex(["Successful", "Failed"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2E86AB", "#E74C3C"]
    bars = ax.bar(counts.index, counts.values, color=colors)
    ax.set_title(_bilingual("Pass/Fail Distribution", "توزيع حالات النجاح والرسوب"))
    ax.set_ylabel(_bilingual("Number of Students", "عدد الطلاب"))
    ax.set_xlabel(_bilingual("Academic Status", "الحالة الأكاديمية"))
    ax.set_xticks(range(len(counts.index)))
    ax.set_xticklabels(
        [
            _bilingual("Successful", "ناجح"),
            _bilingual("Failed", "راسب"),
        ],
        rotation=0,
    )
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    output_path = output_dir / "pass_fail_distribution.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_city_success_rates(
    df: pd.DataFrame, output_dir: Path, top_n: int = 10
) -> Optional[Path]:
    if "city" not in df.columns or df["city"].nunique() == 0:
        return None
    city_stats = (
        df.groupby("city")
        .agg(
            students=("Registration number", "count"),
            pass_rate=("Pass or fail?", lambda x: (x == "Successful").mean() * 100),
        )
        .sort_values(by="students", ascending=False)
        .head(top_n)
    )
    if city_stats.empty:
        return None
    city_stats = city_stats.sort_values(by="pass_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    cities = [ _bidi_text(str(city)) for city in city_stats.index ]
    bars = ax.barh(cities, city_stats["pass_rate"], color="#1ABC9C")
    ax.set_title(
        _bilingual(
            "Top Cities by Student Count and Pass Rate",
            "أعلى المدن من حيث عدد الطلاب ومعدل النجاح",
        )
    )
    ax.set_xlabel(_bilingual("Pass Rate (%)", "نسبة النجاح (%)"))
    ax.set_ylabel(_bilingual("City", "المدينة"))
    for bar, value, count in zip(
        bars, city_stats["pass_rate"], city_stats["students"]
    ):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            _bidi_text(f"{value:.1f}% | {int(count):,} طالب"),
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    output_path = output_dir / "city_success_rates.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_cluster_summary(summary_df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if summary_df.empty:
        return None
    summary_sorted = summary_df.sort_values(by="pass_rate")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    colors = ["#F39C12" if pr < 60 else "#27AE60" for pr in summary_sorted["pass_rate"]]
    bars = ax1.bar(
        summary_sorted["cluster_id"].astype(str),
        summary_sorted["pass_rate"],
        color=colors,
    )
    ax1.set_ylabel(_bilingual("Pass Rate per Cluster (%)", "نسبة النجاح داخل العنقود (%)"))
    ax1.set_xlabel(_bilingual("Cluster ID", "رقم العنقود"))
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.plot(
        summary_sorted["cluster_id"].astype(str),
        summary_sorted["success_rate_mean"],
        color="#2980B9",
        marker="o",
        label=_bilingual("Mean High School Success Rate", "متوسط نسبة النجاح في الثانوية"),
    )
    ax2.set_ylabel(_bilingual("Mean High School Success Rate", "متوسط نسبة النجاح في الثانوية"))
    ax2.set_ylim(0, 100)

    ax1.set_title(
        _bilingual(
            "Cluster Summary: Pass Rates and Student Counts",
            "ملخص العناقيد: نسبة النجاح وعدد الطلاب",
        )
    )
    for bar, students in zip(bars, summary_sorted["students"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            _bidi_text(f"{bar.get_height():.1f}%\n{int(students):,} طالب"),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.95))
    fig.tight_layout()
    output_path = output_dir / "cluster_summary.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_silhouette_scores(clustering_info: dict, output_dir: Path) -> Path:
    scores = clustering_info.get("scores", {})
    if not scores:
        return output_dir / "silhouette_scores.png"
    ks = sorted(scores.keys())
    values = [scores[k] for k in ks]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, values, marker="o", color="#8E44AD")
    ax.set_title(
        _bilingual(
            "Evaluating k using the Silhouette Index",
            "تقييم قيم k باستخدام مؤشر Silhouette",
        )
    )
    ax.set_xlabel(_bilingual("Number of Clusters (k)", "عدد العناقيد (k)"))
    ax.set_ylabel(_bilingual("Silhouette Score", "قيمة مؤشر Silhouette"))
    best_k = clustering_info.get("best_k")
    if best_k is not None:
        ax.axvline(
            x=best_k,
            color="#C0392B",
            linestyle="--",
            label=_bilingual(f"Best k = {best_k}", f"k الأفضل = {best_k}"),
        )
        ax.legend()
    fig.tight_layout()
    output_path = output_dir / "silhouette_scores.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_rules_comparison(
    rules_before_len: int, rules_after_len: int, output_dir: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    categories = [
        _bilingual("Before Clustering", "قبل التجميع"),
        _bilingual("After Clustering", "بعد التجميع"),
    ]
    values = [rules_before_len, rules_after_len]
    bars = ax.bar(categories, values, color=["#7F8C8D", "#16A085"])
    ax.set_title(
        _bilingual(
            "Association Rules Count Comparison",
            "مقارنة عدد القواعد المستخلصة",
        )
    )
    ax.set_ylabel(_bilingual("Number of Rules", "عدد القواعد"))
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    output_path = output_dir / "rules_comparison.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_top_rules_after(
    rules_after: pd.DataFrame, output_dir: Path, top_n: int = 10
) -> Optional[Path]:
    if rules_after.empty:
        return None
    top_rules = rules_after.sort_values(by="lift", ascending=False).head(top_n).copy()
    top_rules["label"] = top_rules.apply(
        lambda row: textwrap.shorten(
            f"ع{row['cluster_id']} | {row['antecedents']} → {row['consequents']}",
            width=90,
            placeholder="…",
        ),
        axis=1,
    )
    top_rules = top_rules.iloc[::-1]  # reverse for better barh ordering
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_rules["label"], top_rules["lift"], color="#34495E")
    ax.set_title(
        _bilingual(
            "Top Post-Clustering Rules (by Lift)",
            "أهم القواعد بعد التجميع (حسب قيمة الرفع)",
        )
    )
    ax.set_xlabel(_bilingual("Lift Value", "قيمة الرفع"))
    for bar, support, confidence in zip(
        bars, top_rules["support"], top_rules["confidence"]
    ):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"support={support:.3f}, confidence={confidence:.2f}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    output_path = output_dir / "top_rules_after.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_recommendations(
    cluster_summary_df: pd.DataFrame, df: pd.DataFrame
) -> List[str]:
    recommendations: List[str] = []
    if cluster_summary_df.empty:
        return recommendations

    high_risk = cluster_summary_df[cluster_summary_df["pass_rate"] < 60]
    high_success = cluster_summary_df[cluster_summary_df["pass_rate"] > 80]

    for _, row in high_risk.iterrows():
        cluster_id_value = row["cluster_id"]
        cluster_id = int(cluster_id_value) if pd.notna(cluster_id_value) else cluster_id_value
        subset = df[df["cluster_id"] == cluster_id]
        common_city = subset["city"].value_counts().idxmax() if not subset["city"].empty else "غير محدد"
        if pd.isna(common_city):
            common_city = "غير محدد"
        recommendations.append(
            f"- التركيز على دعم الطلاب في العنقود {cluster_id} حيث لا تتجاوز نسبة النجاح {row['pass_rate']:.1f}%. أكثر المدن تمثيلاً: {common_city}."
        )

    for _, row in high_success.iterrows():
        cluster_id_value = row["cluster_id"]
        cluster_id = int(cluster_id_value) if pd.notna(cluster_id_value) else cluster_id_value
        subset = df[df["cluster_id"] == cluster_id]
        dominant_spec = subset["Specialization ID"].value_counts().idxmax() if not subset["Specialization ID"].empty else "غير محدد"
        if pd.isna(dominant_spec):
            dominant_spec = "غير محدد"
        recommendations.append(
            f"- دراسة ممارسات العنقود {cluster_id} (نسبة نجاح {row['pass_rate']:.1f}%) مع التركيز على تخصص {dominant_spec} لنقل الخبرات إلى بقية العناقيد."
        )

    recent_registrations = df[df["registration_bucket"] == ">=2015"]
    if not recent_registrations.empty:
        fail_rate_recent = (
            (recent_registrations["Pass or fail?"] == "Failed").mean() * 100
        )
        recommendations.append(
            f"- مراقبة الطلاب المسجلين بعد 2015 حيث تصل نسبة الرسوب ضمنهم إلى {fail_rate_recent:.1f}%."
        )

    return recommendations


def generate_markdown_documentation(
    output_dir: Path,
    total_students: int,
    overall_pass_rate: float,
    apriori_config_before: AprioriConfig,
    clustering_info: dict,
    cluster_summary_df: pd.DataFrame,
    rules_before_len: int,
    rules_after_len: int,
    recommendations: List[str],
    top_rule_before: Optional[pd.Series],
    top_rule_after: Optional[pd.Series],
) -> None:
    lines: List[str] = []
    lines.append("# دراسة استخراج الأنماط وتأثير التجميع على الأداء الأكاديمي")
    lines.append("")
    lines.append("## نظرة عامة على البيانات")
    lines.append(
        f"- عدد السجلات بعد التنظيف: **{total_students:,} طالبًا**."
    )
    lines.append(
        f"- النسبة الكلية للنجاح: **{overall_pass_rate:.1f}%**."
    )
    lines.append("")
    lines.append("![توزيع النجاح والرسوب](pass_fail_distribution.png)")
    lines.append("")
    lines.append("## أبرز المدن أداءً")
    lines.append(
        "يوضح الرسم التالي المدن الأكثر كثافة في قاعدة البيانات مع معدل النجاح لكل مدينة، مما يساعد على تحديد المناطق التي تحتاج إلى تدخل:"
    )
    lines.append("")
    lines.append("![معدلات النجاح حسب المدينة](city_success_rates.png)")
    lines.append("")
    lines.append("## منهجية المعالجة المسبقة")
    lines.append("- إزالة التكرارات وتوحيد صياغة القيم النصية.")
    lines.append("- تعويض السنة الأكاديمية المفقودة بقيمة “Unknown”.")
    lines.append("- إنشاء شرائح زمنية لسنة الميلاد وسنة التسجيل ونسبة النجاح.")
    lines.append("- تجهيز البيانات الترميزية لتغذية خوارزمية Apriori والتجميع.")
    lines.append("")
    lines.append("## نتائج Apriori قبل التجميع")
    lines.append(
        f"- تم استخراج **{rules_before_len} قاعدة** باستخدام دعم أدنى {apriori_config_before.min_support} وثقة أدنى {apriori_config_before.min_confidence}."
    )
    if top_rule_before is not None:
        lines.append(
            f"- أقوى قاعدة قبل التجميع: **{top_rule_before['antecedents']} → {top_rule_before['consequents']}** "
            f"(الدعم {top_rule_before['support']:.3f}، الثقة {top_rule_before['confidence']:.2f}، الرفع {top_rule_before['lift']:.2f})."
        )
    lines.append("")
    lines.append("## اختيار عدد العناقيد")
    lines.append(
        "تم تقييم قيم k من 2 إلى 8 باستخدام مؤشر Silhouette لتحديد البنية التجميعية الأنسب."
    )
    lines.append("")
    lines.append("![تقييم قيم k](silhouette_scores.png)")
    lines.append("")
    lines.append(
        f"- القيمة المثلى: **k = {clustering_info.get('best_k', 'غير محدد')}** مع أعلى معامل Silhouette."
    )
    lines.append("")
    lines.append("## خصائص العناقيد")
    lines.append(
        "يظهر الرسم التالي مقارنة نسب النجاح وعدد الطلاب ومتوسط نسبة النجاح في الثانوية لكل عنقود:"
    )
    lines.append("")
    lines.append("![ملخص العناقيد](cluster_summary.png)")
    lines.append("")
    lines.append("## Apriori بعد التجميع")
    lines.append(
        f"- بعد تقسيم البيانات، تم استخراج **{rules_after_len} قاعدة** أكثر تخصصًا داخل العناقيد."
    )
    lines.append("")
    lines.append("![مقارنة القواعد قبل وبعد](rules_comparison.png)")
    lines.append("")
    lines.append("### أقوى القواعد بعد التجميع")
    if top_rule_after is not None:
        lines.append(
            f"- أقوى قاعدة بعد التجميع (رفع {top_rule_after['lift']:.2f}): "
            f"عنقود {int(top_rule_after['cluster_id'])} | {top_rule_after['antecedents']} → {top_rule_after['consequents']}."
        )
    lines.append("")
    lines.append("![أهم القواعد بعد التجميع](top_rules_after.png)")
    lines.append("")
    lines.append("## التوصيات العملية")
    if recommendations:
        lines.extend(recommendations)
    else:
        lines.append("- لم يتم تحديد توصيات مباشرة من البيانات الحالية.")
    lines.append("")
    lines.append("## الخلاصة النهائية")
    lines.append(
        "- تطبيق التجميع قبل استخراج القواعد أظهر أن الأنماط أصبحت أكثر دقة ومرتبطة بشرائح طلابية محددة، "
        "مما يوفر أساسًا أقوى لاتخاذ قرارات موجهة."
    )
    lines.append(
        "- يوصى بتحديث البيانات دوريًا وإعادة تنفيذ التحليل لمراقبة التغيرات واستباق التحديات الأكاديمية."
    )
    lines.append("")
    lines.append("## كيفية إعادة التنفيذ")
    lines.append("```bash")
    lines.append("python analysis_pipeline.py")
    lines.append("```")

    with (output_dir / "documentation.md").open("w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))

def main():
    base_path = Path(__file__).resolve().parent
    data_path = base_path / "high_school.csv"
    output_dir = base_path / "outputs"

    ensure_output_dir(output_dir)

    raw_df = load_data(data_path)
    df = preprocess(raw_df)

    apriori_features = [
        "Gender",
        "city",
        "Secondary system",
        "School system",
        "Specialization ID",
        "Pass or fail?",
        "Success sign",
        "Academic year",
        "birth_bucket",
        "registration_bucket",
        "success_rate_band",
    ]

    encoded_transactions = encode_for_apriori(df, apriori_features)
    apriori_config_before = AprioriConfig(min_support=0.05, min_confidence=0.6, max_rules=30)
    rules_before = run_apriori(encoded_transactions, apriori_config_before)
    save_rules(rules_before, output_dir / "apriori_rules_before.csv")

    feature_matrix = prepare_features_for_clustering(df)
    cluster_labels, clustering_info = perform_clustering(feature_matrix)
    df["cluster_id"] = cluster_labels
    df.to_csv(output_dir / "students_with_clusters.csv", index=False)

    summary = summarise_clusters(df, "cluster_id")
    summary.to_csv(output_dir / "cluster_summary.csv", index=False)

    with (output_dir / "kmeans_silhouette_scores.json").open("w", encoding="utf-8") as fp:
        json.dump(clustering_info, fp, indent=2)

    # Run Apriori within each cluster
    cluster_rules_frames: List[pd.DataFrame] = []
    for cluster_value, cluster_df in df.groupby("cluster_id"):
        encoded_cluster = encode_for_apriori(cluster_df, apriori_features)
        dynamic_support = max(0.02, 50 / len(cluster_df))
        cluster_config = AprioriConfig(
            min_support=dynamic_support,
            min_confidence=0.6,
            max_rules=20,
        )
        cluster_rules = run_apriori(encoded_cluster, cluster_config)
        if cluster_rules.empty:
            continue
        cluster_rules.insert(0, "cluster_id", cluster_value)
        cluster_rules_frames.append(cluster_rules)

    if cluster_rules_frames:
        rules_after = pd.concat(cluster_rules_frames, ignore_index=True)
    else:
        rules_after = pd.DataFrame()

    save_rules(rules_after, output_dir / "apriori_rules_after_clusters.csv")

    # Generate visualisations
    plot_pass_fail_distribution(df, output_dir)
    plot_city_success_rates(df, output_dir)
    plot_cluster_summary(summary, output_dir)
    plot_silhouette_scores(clustering_info, output_dir)
    plot_rules_comparison(len(rules_before), len(rules_after), output_dir)
    plot_top_rules_after(rules_after, output_dir)

    # Generate recommendations
    recommendations = generate_recommendations(summary, df)

    # Generate summary text
    summary_lines = []
    summary_lines.append("Data preprocessing steps:")
    summary_lines.append("- Dropped duplicate records and standardised column formats.")
    summary_lines.append("- Imputed missing academic years with 'Unknown' and engineered buckets for birth year, registration period, and success rate.")
    summary_lines.append("\nApriori before clustering:")
    summary_lines.append(
        f"- Evaluated {encoded_transactions.shape[0]} transactions across {encoded_transactions.shape[1]} item indicators."
    )
    summary_lines.append(
        f"- Discovered {len(rules_before)} rules with minimum support {apriori_config_before.min_support} and confidence {apriori_config_before.min_confidence}."
    )

    summary_lines.append("\nK-Means clustering:")
    summary_lines.append(
        f"- Assessed k values 2-8; selected k={clustering_info['best_k']} with best silhouette score {clustering_info['scores'][clustering_info['best_k']]:.3f}."
    )
    summary_lines.append(
        "- Saved cluster assignments with aggregated pass-rate and success-rate statistics."
    )

    summary_lines.append("\nApriori after clustering:")
    if rules_after.empty:
        summary_lines.append("- No rules met the support/confidence thresholds within individual clusters.")
    else:
        summary_lines.append(
            f"- Generated {len(rules_after)} cluster-specific rules using adaptive support >= {min(r['support'] for _, r in rules_after.iterrows()):.3f}."
        )
        summary_lines.append("- Stored combined results in apriori_rules_after_clusters.csv for comparison.")
        summary_lines.append("\nKey recommendations based on clusters:")
        for rec in recommendations:
            summary_lines.append(rec)

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as fp:
        fp.write("\n".join(summary_lines))

    # Generate detailed markdown documentation
    total_students = len(df)
    overall_pass_rate = (df["Pass or fail?"] == "Successful").mean() * 100
    top_rule_before = rules_before.iloc[0] if not rules_before.empty else None
    top_rule_after = (
        rules_after.sort_values(by="lift", ascending=False).iloc[0]
        if not rules_after.empty
        else None
    )
    generate_markdown_documentation(
        output_dir=output_dir,
        total_students=total_students,
        overall_pass_rate=overall_pass_rate,
        apriori_config_before=apriori_config_before,
        clustering_info=clustering_info,
        cluster_summary_df=summary,
        rules_before_len=len(rules_before),
        rules_after_len=len(rules_after),
        recommendations=recommendations,
        top_rule_before=top_rule_before,
        top_rule_after=top_rule_after,
    )


if __name__ == "__main__":
    main()

