"""
Mandarin tone sandhi rules.

Tone sandhi refers to the systematic changes in tone that occur when
certain tones appear in sequence. This module applies these rules to
convert underlying/lexical tones to surface/spoken tones.

Main rules implemented:
1. Third tone sandhi: T3 + T3 → T2 + T3
2. 不 (bù) sandhi: T4 → T2 before T4
3. 一 (yī) sandhi: T1 → T4 before T1/T2/T3, T1 → T2 before T4
4. Half-third tone: T3 before non-T3 → half-third (T3 for evaluation)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SandhiResult:
    """Result of sandhi application."""
    spoken_tones: list[int]
    rules_applied: list[str]


def apply_sandhi(
    tones: list[int],
    pinyin: Optional[list[str]] = None,
    characters: Optional[str] = None,
) -> SandhiResult:
    """
    Apply tone sandhi rules to get spoken tones.

    Args:
        tones: List of underlying/lexical tone numbers (1-5)
        pinyin: Optional list of pinyin syllables (for 不/一 detection)
        characters: Optional Chinese characters (for 不/一 detection)

    Returns:
        SandhiResult with spoken tones and list of rules applied
    """
    if not tones:
        return SandhiResult(spoken_tones=[], rules_applied=[])

    spoken = list(tones)  # Copy
    rules_applied = []

    # Apply 不 sandhi first (before T3 sandhi to handle 不不 correctly)
    spoken, bu_rules = _apply_bu_sandhi(spoken, pinyin, characters)
    rules_applied.extend(bu_rules)

    # Apply 一 sandhi
    spoken, yi_rules = _apply_yi_sandhi(spoken, pinyin, characters)
    rules_applied.extend(yi_rules)

    # Apply third tone sandhi (most common)
    spoken, t3_rules = _apply_third_tone_sandhi(spoken)
    rules_applied.extend(t3_rules)

    return SandhiResult(spoken_tones=spoken, rules_applied=rules_applied)


def _apply_third_tone_sandhi(tones: list[int]) -> tuple[list[int], list[str]]:
    """
    Apply third tone sandhi rule.

    Rule: When two or more T3 syllables occur in sequence,
    all but the last become T2.

    Examples:
        T3 + T3 → T2 + T3 (你好 nǐ hǎo → ní hǎo)
        T3 + T3 + T3 → T2 + T2 + T3 (我很好 wǒ hěn hǎo → wó hén hǎo)
    """
    result = list(tones)
    rules = []

    n = len(result)
    i = 0

    while i < n:
        # Find a sequence of consecutive T3s
        if result[i] == 3:
            # Count how many T3s in a row
            j = i
            while j < n and result[j] == 3:
                j += 1

            seq_len = j - i

            if seq_len >= 2:
                # Change all but the last T3 to T2
                for k in range(i, j - 1):
                    result[k] = 2
                    rules.append(f"T3→T2 at position {k} (third tone sandhi)")

            i = j
        else:
            i += 1

    return result, rules


def _apply_bu_sandhi(
    tones: list[int],
    pinyin: Optional[list[str]],
    characters: Optional[str],
) -> tuple[list[int], list[str]]:
    """
    Apply 不 (bù) sandhi rule.

    Rule: 不 (T4) becomes T2 when followed by another T4.

    Examples:
        不对 bù duì → bú duì
        不是 bù shì → bú shì
    """
    result = list(tones)
    rules = []

    if not pinyin and not characters:
        return result, rules

    n = len(result)

    for i in range(n - 1):
        is_bu = False

        # Check if this syllable is 不
        if characters and i < len(characters) and characters[i] == '不':
            is_bu = True
        elif pinyin and i < len(pinyin) and pinyin[i].lower() in ('bu', 'bù', 'bú'):
            is_bu = True

        if is_bu and result[i] == 4 and result[i + 1] == 4:
            result[i] = 2
            rules.append(f"不 T4→T2 at position {i} (before T4)")

    return result, rules


def _apply_yi_sandhi(
    tones: list[int],
    pinyin: Optional[list[str]],
    characters: Optional[str],
) -> tuple[list[int], list[str]]:
    """
    Apply 一 (yī) sandhi rules.

    Rules:
        - 一 (T1) → T4 before T1, T2, or T3
        - 一 (T1) → T2 before T4

    Examples:
        一天 yī tiān → yì tiān (before T1)
        一年 yī nián → yì nián (before T2)
        一起 yī qǐ → yì qǐ (before T3)
        一个 yī gè → yí gè (before T4)

    Note: 一 keeps T1 when:
        - In isolation
        - At the end of a word
        - When counting/listing (第一, 一二三)
    """
    result = list(tones)
    rules = []

    if not pinyin and not characters:
        return result, rules

    n = len(result)

    for i in range(n - 1):  # Not the last syllable
        is_yi = False

        # Check if this syllable is 一
        if characters and i < len(characters) and characters[i] == '一':
            is_yi = True
        elif pinyin and i < len(pinyin) and pinyin[i].lower() in ('yi', 'yī', 'yí', 'yì'):
            is_yi = True

        if is_yi and result[i] == 1:
            next_tone = result[i + 1]

            if next_tone == 4:
                # Before T4: 一 becomes T2
                result[i] = 2
                rules.append(f"一 T1→T2 at position {i} (before T4)")
            elif next_tone in (1, 2, 3):
                # Before T1/T2/T3: 一 becomes T4
                result[i] = 4
                rules.append(f"一 T1→T4 at position {i} (before T{next_tone})")

    return result, rules


def get_acceptable_tones(
    expected_tone: int,
    position: int,
    all_tones: list[int],
    pinyin: Optional[list[str]] = None,
    characters: Optional[str] = None,
) -> set[int]:
    """
    Get the set of acceptable tones for a position, considering sandhi.

    This is useful for evaluation: we accept EITHER the underlying tone
    OR the sandhi-modified tone as correct.

    Args:
        expected_tone: The underlying/lexical tone at this position
        position: Index of the syllable
        all_tones: All tones in the phrase
        pinyin: Optional pinyin list
        characters: Optional characters

    Returns:
        Set of acceptable tone numbers
    """
    acceptable = {expected_tone}

    # Apply sandhi to get spoken tone
    sandhi_result = apply_sandhi(all_tones, pinyin, characters)
    spoken_tone = sandhi_result.spoken_tones[position]

    # Accept both underlying and spoken tones
    acceptable.add(spoken_tone)

    # Special case: T3 before non-T3 can be pronounced as "half third"
    # which might be detected as T2 or T3
    if expected_tone == 3 and position < len(all_tones) - 1:
        next_tone = all_tones[position + 1]
        if next_tone != 3:
            # Half-third tone context - accept T2 as well
            acceptable.add(2)

    return acceptable


def describe_sandhi(
    tones: list[int],
    pinyin: Optional[list[str]] = None,
    characters: Optional[str] = None,
) -> str:
    """
    Get a human-readable description of sandhi application.

    Useful for educational feedback to users.
    """
    result = apply_sandhi(tones, pinyin, characters)

    if not result.rules_applied:
        return "No tone sandhi applies to this phrase."

    lines = ["Tone sandhi rules applied:"]

    for rule in result.rules_applied:
        lines.append(f"  • {rule}")

    # Show before/after
    underlying = " ".join(f"T{t}" for t in tones)
    spoken = " ".join(f"T{t}" for t in result.spoken_tones)
    lines.append(f"\nUnderlying: {underlying}")
    lines.append(f"Spoken:     {spoken}")

    return "\n".join(lines)


# Common phrases with sandhi for reference/testing
SANDHI_EXAMPLES = {
    "你好": {
        "pinyin": ["nǐ", "hǎo"],
        "underlying": [3, 3],
        "spoken": [2, 3],
        "rule": "third tone sandhi",
    },
    "我很好": {
        "pinyin": ["wǒ", "hěn", "hǎo"],
        "underlying": [3, 3, 3],
        "spoken": [2, 2, 3],
        "rule": "third tone sandhi (chain)",
    },
    "不对": {
        "pinyin": ["bù", "duì"],
        "underlying": [4, 4],
        "spoken": [2, 4],
        "rule": "不 sandhi",
    },
    "不好": {
        "pinyin": ["bù", "hǎo"],
        "underlying": [4, 3],
        "spoken": [4, 3],
        "rule": "no sandhi (不 before T3)",
    },
    "一个": {
        "pinyin": ["yī", "gè"],
        "underlying": [1, 4],
        "spoken": [2, 4],
        "rule": "一 sandhi (before T4)",
    },
    "一天": {
        "pinyin": ["yī", "tiān"],
        "underlying": [1, 1],
        "spoken": [4, 1],
        "rule": "一 sandhi (before T1)",
    },
    "小姐": {
        "pinyin": ["xiǎo", "jiě"],
        "underlying": [3, 3],
        "spoken": [2, 3],
        "rule": "third tone sandhi",
    },
    "可以": {
        "pinyin": ["kě", "yǐ"],
        "underlying": [3, 3],
        "spoken": [2, 3],
        "rule": "third tone sandhi",
    },
}


if __name__ == "__main__":
    # Test sandhi rules
    print("Testing tone sandhi rules\n")
    print("=" * 50)

    for chars, data in SANDHI_EXAMPLES.items():
        result = apply_sandhi(
            data["underlying"],
            data["pinyin"],
            chars,
        )

        expected = data["spoken"]
        passed = result.spoken_tones == expected

        status = "✓" if passed else "✗"
        print(f"\n{status} {chars} ({' '.join(data['pinyin'])})")
        print(f"  Underlying: {data['underlying']}")
        print(f"  Expected:   {expected}")
        print(f"  Got:        {result.spoken_tones}")
        if result.rules_applied:
            print(f"  Rules:      {result.rules_applied}")

    print("\n" + "=" * 50)
    print("\nDescriptive output for 你好:")
    print(describe_sandhi([3, 3], ["nǐ", "hǎo"], "你好"))
