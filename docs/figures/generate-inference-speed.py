"""Regenerate the bilingual SVGs from the published MFQ benchmark snapshot.

uv run --no-project --with matplotlib python docs/figures/generate-inference-speed.py
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "data/qwen38-flash-next-inference-speed.json").read_text())

# Register the local CJK fonts when available. SVG glyphs are saved as paths,
# so the exported files do not depend on the viewer having these fonts.
font_family = "DejaVu Sans"
for font_path in (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
):
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        font_family = font_manager.FontProperties(fname=font_path).get_name()

plt.rcParams.update(
    {
        "font.family": font_family,
        "font.size": 11,
        "svg.fonttype": "path",
        "svg.hashsalt": "mfq-flash-next-inference-speed",
        "axes.unicode_minus": False,
        "axes.edgecolor": "#dce3ec",
        "text.color": "#203149",
        "xtick.color": "#64758c",
        "ytick.color": "#30425a",
    }
)

COLORS = {"plain": "#68839f", "mtp": "#ac783b", "prefill": "#42877f"}
LANGUAGES = {
    "en": {
        "title": "Inference speed",
        "subtitle": "Qwen3.8-Flash-Next · 4,096-token input · Published MFQ tiers",
        "generation": "Generation (tok/s)",
        "prefill": "Prefill (tok/s)",
        "plain": "MTP off",
        "mtp": "MTP on",
        "metal": "M5 Max · Metal",
        "cuda": "RTX 3090 Ti · CUDA",
        "footnote2": "Generation bars compare MTP settings. Hardware and RAM configurations are part of each result.",
    },
    "zh-CN": {
        "title": "推理速度",
        "subtitle": "Qwen3.8-Flash-Next · 输入 4,096 tokens · 已发布 MFQ 档位",
        "generation": "生成速度（tok/s）",
        "prefill": "提示词处理（tok/s）",
        "plain": "关闭 MTP",
        "mtp": "开启 MTP",
        "metal": "M5 Max · Metal",
        "cuda": "RTX 3090 Ti · CUDA",
        "footnote2": "生成速度的两根柱分别对应 MTP 开关；硬件与内存配置是各项结果的一部分。",
    },
}


def panel(fig, bounds, backend, metric):
    ax = fig.add_axes(bounds)
    rows = backend["measurements"]
    is_generation = metric == "generation"
    limit = 120 if is_generation else 3000
    ticks = range(0, 121, 20) if is_generation else (0, 1000, 2000, 3000)

    for i, row in enumerate(rows):
        if row["tier"] == "V3-XXS":
            ax.axhspan(i - 0.48, i + 0.48, color="#f6eee1", zorder=0)
        if is_generation:
            for offset, key, color in (
                (-0.17, "generation_no_mtp", COLORS["plain"]),
                (0.17, "generation_mtp", COLORS["mtp"]),
            ):
                value = row[key]
                bar = ax.barh(i + offset, value, height=0.27, color=color, zorder=3)[0]
                bar.set_gid(f"{backend['id']}-{row['tier']}-{key}")
                ax.text(value + 1.3, i + offset, f"{value:.1f}", va="center", fontsize=11.5, color=color)
        else:
            value = row["prefill"]
            bar = ax.barh(i, value, height=0.46, color=COLORS["prefill"], zorder=3)[0]
            bar.set_gid(f"{backend['id']}-{row['tier']}-prefill")
            ax.text(value + 38, i, f"{value:,.1f}", va="center", fontsize=11.5, color="#326f69")

    labels = [row["tier"] for row in rows]
    ax.set_yticks(range(len(rows)), labels, fontsize=11.5)
    for tick, row in zip(ax.get_yticklabels(), rows):
        if row["tier"] == "V3-XXS":
            tick.set_fontweight("bold")
            tick.set_color("#805b2f")
    ax.set_xlim(0, limit)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_ylim(len(rows) - 0.55, -0.55)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e6ebf1", linewidth=0.8)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", length=0, pad=8, labelsize=10.5)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    return ax


def generate(locale, text):
    fig = plt.figure(figsize=(14, 10.6), facecolor="white")
    fig.text(0.065, 0.96, text["title"], fontsize=25, weight="bold")
    fig.text(0.065, 0.905, text["generation"], fontsize=15, weight="bold")
    fig.legend(
        [Patch(color=COLORS["plain"]), Patch(color=COLORS["mtp"])],
        [text["plain"], text["mtp"]],
        loc="center right", bbox_to_anchor=(0.975, 0.965), ncol=2,
        frameon=False, fontsize=11.5, handlelength=1.8, columnspacing=2,
    )
    fig.text(0.065, 0.445, text["prefill"], fontsize=15, weight="bold")

    for left, backend in zip((0.08, 0.585), DATA["backends"]):
        fig.text(left, 0.855, text[backend["id"]], fontsize=14, weight="bold")
        panel(fig, (left, 0.52, 0.375, 0.30), backend, "generation")
        panel(fig, (left, 0.08, 0.375, 0.305), backend, "prefill")

    suffix = "" if locale == "en" else ".zh-CN"
    target = ROOT / f"qwen38-flash-next-inference-speed{suffix}.svg"
    fig.savefig(target, metadata={"Title": text["title"], "Description": text["subtitle"], "Creator": "MFQ", "Date": None})
    svg = target.read_text()
    svg = svg.replace("<svg ", '<svg role="img" aria-labelledby="title desc" ', 1)
    svg = svg.replace(" <title>", ' <title id="title">', 1)
    svg = svg.replace(" <metadata>", f' <desc id="desc">{escape(text["subtitle"] + ". " + text["footnote2"])}</desc>\n <metadata>', 1)
    target.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    preview = Path(f"/tmp/mfq-inference-speed{suffix or '.en'}.png")
    fig.savefig(preview, dpi=120, facecolor="white")
    plt.close(fig)
    print(target)
    print(preview)


if __name__ == "__main__":
    for locale, text in LANGUAGES.items():
        generate(locale, text)
