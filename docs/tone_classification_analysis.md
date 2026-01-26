# Tone Classification Accuracy Analysis

Analysis performed: 2026-01-26
Updated: 2026-01-26 (Root cause identified)

## Overview

This document summarizes the accuracy of the tone classifier when evaluated against TTS-generated minimal pairs (words that differ only in tone).

## Executive Summary

**Root cause identified: F0 features are hurting T3 classification.**

| Configuration | T3 Accuracy | Overall |
|--------------|-------------|---------|
| MMS features only | **100%** | 100% |
| MMS + F0 features | **66.7%** | 91.7% |
| Current production model | **0%** | 73.3% |

The MMS encoder alone can perfectly classify all tones. F0 fusion introduces confusion for Tone 3.

## Test Methodology

- **Test samples**: 30 single-character words across 8 minimal pair groups (ma, shi, shu, mai, tang, wen, yi, du)
- **TTS backend**: LightTTS (edge-tts) with zh-CN-XiaoxiaoNeural voice
- **Evaluation modes**:
  1. Classifier-only (no ASR)
  2. Full evaluator (classifier + MMS ASR with OR logic)

## Results Summary

| Metric | Classifier Only | With ASR |
|--------|-----------------|----------|
| Overall Accuracy | 73.3% (22/30) | 76.7% (23/30) |

### Accuracy by Tone

| Tone | Description | Accuracy | Notes |
|------|-------------|----------|-------|
| T1 | High level | **100%** (7/7) | Perfect detection |
| T2 | Rising | **100%** (7/7) | Perfect detection |
| T3 | Dipping | **0%** (0/8) | Complete failure |
| T4 | Falling | **100%** (8/8) | Perfect detection |

### Confusion Matrix (Classifier Only)

```
        Predicted
        T1  T2  T3  T4
    T1   7   0   0   0
E   T2   0   7   0   0
x   T3   3   5   0   0  <-- Never predicts T3
p   T4   0   0   0   8
```

**Key observation**: The classifier never predicts Tone 3. All T3 samples are misclassified as either T1 (3 cases) or T2 (5 cases).

### ASR Override Analysis

| Scenario | Count |
|----------|-------|
| Both correct | 22 |
| Classifier wrong, ASR saved | 1 |
| Classifier right, ASR broke | 0 |
| Both wrong | 7 |

The ASR override logic provides marginal improvement (+1 case), primarily because ASR also struggles with T3 recognition on single-character utterances.

### Classifier Confidence

| Outcome | Average Confidence | Range |
|---------|-------------------|-------|
| Correct predictions | 98.0% | 94.1% - 100.0% |
| Wrong predictions | 80.7% | 53.1% - 95.8% |

The classifier maintains high confidence even on incorrect predictions, indicating systematic bias rather than uncertainty.

## Analysis

### Why Tone 3 Fails

Several factors may contribute to the complete failure on Tone 3:

1. **Training data distribution**: T3 may be underrepresented or have labeling issues in the training set

2. **Acoustic similarity**: T3's "dipping" contour (falling then rising) shares characteristics with:
   - T2 (rising) - the final rise of T3 resembles T2
   - T1 (high level) - short T3 utterances may not complete the full dip

3. **TTS generation**: Edge-tts may not produce canonical T3 contours for single characters. In natural speech, T3 is often realized as a low tone (half-third) before other syllables.

4. **Context dependency**: T3 is the most context-dependent tone in Mandarin due to sandhi rules. Isolated T3 may have inconsistent acoustic realization.

### Accuracy by Pinyin Base

All pinyin bases show identical 75% accuracy (3/4 correct), with the single failure always being the T3 variant:

| Pinyin | Accuracy | Failed Character |
|--------|----------|------------------|
| ma | 75% | 马 (T3) |
| shi | 75% | 史 (T3) |
| shu | 75% | 鼠 (T3) |
| mai | 50% | 买 (T3) |
| tang | 75% | 躺 (T3) |
| wen | 75% | 吻 (T3) |
| yi | 75% | 椅 (T3) |
| du | 75% | 堵 (T3) |

## Root Cause Analysis

### Layer Analysis

We performed a systematic analysis of all 25 MMS layers (including embeddings) to determine if our layer selection was suboptimal.

**Finding: Layer selection is NOT the problem.**

| Layer | Accuracy | Notes |
|-------|----------|-------|
| Layer 17 | 98.3% | Best single layer |
| Layer 3 | 96.7% | Strong early layer |
| Layers 16-22 | 95-97% | All high accuracy |
| Current config (8-12, 15-17) | 96.7% | Competitive |

All layers from 0-23 achieve 90%+ accuracy with simple linear probes, with **100% T3 accuracy**.

### F0 Feature Analysis

We then tested whether F0 features were the problem by retraining with clean TTS data:

| Configuration | T1 | T2 | T3 | T4 | Overall |
|--------------|----|----|----|----|---------|
| MMS only | 100% | 100% | **100%** | 100% | 100% |
| MMS + F0 | 100% | 100% | **66.7%** | 100% | 91.7% |

**Conclusion: F0 features are causing T3 misclassification as T2.**

### Why F0 Hurts T3

Tone 3 has a distinctive "dipping" F0 contour (high → low → rise). However:

1. **Short audio clips** may not capture the full T3 contour
2. **The final rise** of T3 can resemble T2 (rising tone)
3. **F0 summary statistics** (mean, range, slope) lose the temporal shape
4. The model learns to rely on F0 signals that are **misleading for T3**

## Recommendations

### Immediate Fix (Recommended)

**Disable F0 features for the tone classifier.**

The MMS encoder already captures prosodic information in layers 8-12. The F0 fusion adds complexity without benefit and actively harms T3 detection.

```python
# In evaluator.py or during model loading
evaluator = Evaluator(use_f0=False)
```

Or retrain the model with `use_f0=False`:

```bash
python -m src.model.train --use-f0 false ...
```

### Alternative Improvements

1. **Improve F0 feature representation**: Instead of summary statistics, use the actual F0 contour shape (e.g., via learned embeddings or contour templates)

2. **T3-specific F0 handling**: Only use F0 for T1/T2/T4 discrimination, disable for T3

3. **Contour-based classification**: Match F0 contours against tone templates using DTW or learned similarity

## Test Reproduction

To run this analysis:

```bash
source venv/bin/activate
python tests/test_tone_classification_accuracy.py
```

The test generates TTS audio for minimal pairs and evaluates both classifier-only and full evaluator accuracy.

## Appendix: Test Cases

### Minimal Pairs Used

| Pinyin | T1 | T2 | T3 | T4 |
|--------|----|----|----|----|
| ma | 妈 (mother) | 麻 (hemp) | 马 (horse) | 骂 (scold) |
| shi | 诗 (poem) | 十 (ten) | 史 (history) | 是 (is) |
| shu | 书 (book) | 熟 (cooked) | 鼠 (rat) | 树 (tree) |
| mai | - | - | 买 (buy) | 卖 (sell) |
| tang | 汤 (soup) | 糖 (sugar) | 躺 (lie down) | 烫 (hot) |
| wen | 温 (warm) | 文 (culture) | 吻 (kiss) | 问 (ask) |
| yi | 一 (one) | 姨 (aunt) | 椅 (chair) | 意 (meaning) |
| du | 都 (capital) | 读 (read) | 堵 (block) | 度 (degree) |
