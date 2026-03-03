import unicodedata
import re
from collections import Counter

# === COPY YOUR CLEAN FUNCTION EXACTLY ===
def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    # text = text.lower()

    text = re.sub(r"_\w+", "", text)
    text = re.sub(r"<.*?>", "", text)

    text = "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("C")
    )

    text = re.sub(r"\s+", " ", text)

    return text


file_path = "intro-to-nlp-project/data/zh.txt"

with open(file_path, "r", encoding="utf8", errors="ignore") as f:
    raw_lines = [line.rstrip("\n") for line in f]

cleaned_lines = [clean_text(line).strip() for line in raw_lines if clean_text(line).strip()]

print("===== BASIC STATS =====")
print("Raw lines:", len(raw_lines))
print("Cleaned non-empty lines:", len(cleaned_lines))

lengths = [len(line) for line in cleaned_lines]
print("Average cleaned line length:", sum(lengths)/len(lengths))
print("Median cleaned line length:", sorted(lengths)[len(lengths)//2])


print("\n===== SCRIPT DISTRIBUTION =====")

script_counts = Counter()

for line in cleaned_lines:
    for ch in line:
        name = unicodedata.name(ch, "")
        if "HIRAGANA" in name:
            script_counts["hiragana"] += 1
        elif "KATAKANA" in name:
            script_counts["katakana"] += 1
        elif "CJK UNIFIED" in name:
            script_counts["kanji"] += 1
        elif "LATIN" in name:
            script_counts["latin"] += 1
        elif ch.isdigit():
            script_counts["digits"] += 1
        elif ch.isspace():
            continue
        else:
            script_counts["other"] += 1

print(script_counts)


print("\n===== DUPLICATION =====")

unique_lines = len(set(cleaned_lines))
print("Unique cleaned lines:", unique_lines)
print("Duplicate cleaned lines:", len(cleaned_lines) - unique_lines)


print("\n===== NON-JAPANESE HEAVY LINES =====")

def japanese_ratio(line):
    jp = 0
    total = 0
    for ch in line:
        if ch.isspace():
            continue
        name = unicodedata.name(ch, "")
        if ("HIRAGANA" in name or
            "KATAKANA" in name or
            "CJK UNIFIED" in name):
            jp += 1
        total += 1
    return jp / total if total > 0 else 0


low_jp_lines = [line for line in cleaned_lines if japanese_ratio(line) < 0.5]

print("Lines with <50% Japanese chars:", len(low_jp_lines))

print("\nSample problematic lines:")
for l in low_jp_lines[:20]:
    print(l)


print("\n===== VERY SHORT LINES =====")
short_lines = [l for l in cleaned_lines if len(l) <= 3]
print("Lines length <=3:", len(short_lines))

print("\nSample short lines:")
for l in short_lines[:20]:
    print(l)