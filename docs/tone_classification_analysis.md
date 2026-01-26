# Tone Classification Accuracy Analysis

Analysis performed: 2026-01-26

## Overview

This document summarizes the accuracy of the tone classifier when evaluated against TTS-generated minimal pairs (words that differ only in tone).

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

## Recommendations

### Short-term Mitigations

1. **Increase ASR weight for T3**: When classifier predicts T1/T2 with moderate confidence, give more weight to ASR tone detection

2. **Add T3-specific heuristics**: Use F0 contour analysis to detect the characteristic dip pattern

3. **Lower threshold for T3 sandhi**: Accept T2 pronunciations when T3 is expected (natural sandhi behavior)

### Long-term Improvements

1. **Retrain classifier with balanced T3 data**: Ensure training set has sufficient, correctly-labeled T3 samples

2. **Use T3-aware augmentation**: Apply pitch transformations that preserve T3 characteristics

3. **Multi-TTS training**: Train on audio from multiple TTS engines to improve generalization

4. **Real speech validation**: Test against human recordings to verify this isn't a TTS artifact

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
