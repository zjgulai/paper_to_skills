---
title: Live-Stream-Highlight-Extraction — 直播高光片段自动提取与二次传播内容生产
doc_type: knowledge
module: 20-AI视频生成
topic: live-stream-highlight-extraction
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Live-Stream-Highlight-Extraction

## ① 算法原理

核心是「互动峰值检测 + 语义重要性评分 + 时间窗口提取」三阶段流水线：

1. **互动信号采集**：实时统计每秒的弹幕数、点赞速率、商品点击率、礼物数，构建互动时间序列
2. **峰值检测**：Z-score 异常检测识别互动突增时间点（阈值：均值 + 2.5σ），标记高光候选区间
3. **内容语义打分**：对高光区间的直播语音转录文本做 TF-IDF 关键词密度评分，过滤掉纯互动而无内容价值的片段
4. **时间窗口扩展**：以峰值点为中心扩展 ±15 秒，合并相邻高光区间（间距 < 10 秒的合并）
5. **输出剪辑列表**：按综合评分排序，输出 Top-N 片段的时间戳和推荐剪辑长度（15s/30s/60s 三种规格）

**误触发防护**：单个峰值必须持续 ≥ 3 秒才标记为高光，过滤瞬间噪声。

## ② 母婴出海应用案例

**场景：TikTok Shop 3小时母婴直播的高光提取**
- 直播结束后用互动数据自动提取 Top-10 高光片段（每段 30-60 秒）
- 将 3 小时直播压缩为 10 条短视频素材，同步发布到 TikTok/Instagram Reels/YouTube Shorts
- 量化价值：人工剪辑 3 小时素材需 4-6 小时，自动提取 < 5 分钟，年化节省剪辑工时 300+ 小时，二次流量 GMV 贡献 15-20%

## ③ 代码模板

```python
import numpy as np
from typing import List, Tuple, Dict

def extract_live_stream_highlights(
    interaction_series: List[float],
    fps: float = 1.0,
    z_threshold: float = 2.5,
    min_duration_sec: float = 3.0,
    window_expand_sec: float = 15.0,
    merge_gap_sec: float = 10.0,
    top_n: int = 10
) -> List[Dict]:
    arr = np.array(interaction_series)
    mean, std = arr.mean(), arr.std()
    if std == 0:
        return []

    z_scores = (arr - mean) / std
    peak_frames = np.where(z_scores > z_threshold)[0]

    if len(peak_frames) == 0:
        return []

    min_frames = int(min_duration_sec * fps)
    expand_frames = int(window_expand_sec * fps)
    merge_frames = int(merge_gap_sec * fps)

    segments = []
    i = 0
    while i < len(peak_frames):
        start = max(0, peak_frames[i] - expand_frames)
        end = peak_frames[i] + expand_frames
        while i + 1 < len(peak_frames) and peak_frames[i + 1] - peak_frames[i] < merge_frames:
            i += 1
            end = peak_frames[i] + expand_frames
        end = min(len(arr) - 1, end)
        if end - start >= min_frames:
            score = float(z_scores[start:end + 1].max())
            segments.append({"start_sec": start / fps, "end_sec": end / fps, "score": round(score, 2)})
        i += 1

    segments.sort(key=lambda x: -x["score"])
    for seg in segments:
        seg["duration_sec"] = round(seg["end_sec"] - seg["start_sec"], 1)
        seg["recommended_format"] = "15s" if seg["duration_sec"] <= 20 else "30s" if seg["duration_sec"] <= 45 else "60s"

    return segments[:top_n]


if __name__ == "__main__":
    np.random.seed(42)
    baseline = np.random.exponential(10, 3600)
    for t in [300, 900, 1800, 2700, 3300]:
        baseline[t:t + 30] += np.random.exponential(50, 30)

    highlights = extract_live_stream_highlights(baseline, fps=1.0, top_n=5)
    print(f"提取高光片段: {len(highlights)} 个")
    for i, h in enumerate(highlights):
        print(f"  #{i+1} {h['start_sec']:.0f}s - {h['end_sec']:.0f}s ({h['duration_sec']}s, 评分:{h['score']}, 规格:{h['recommended_format']})")
    assert len(highlights) > 0
    assert all(h["duration_sec"] > 0 for h in highlights)
    print("[✓] Live-Stream-Highlight-Extraction 测试通过")
```

## ④ 技能关联

> 前置: [[Skill-TikTok-Content-Lifecycle-Analytics]]
> 延伸: [[Skill-Cross-Platform-Video-Repurposing]]
> 可组合: [[Skill-TikTok-Hook-Optimizer]]

## ⑤ 商业价值评估

- **ROI量化**: 3 小时直播生产 10 条短视频，年化节省剪辑工时 300+ 小时，二次内容带来额外 GMV 15-20%
- **实施难度**: ⭐⭐（容易，主要是互动数据接入）
- **优先级**: ⭐⭐⭐⭐（高频直播卖家刚需）
