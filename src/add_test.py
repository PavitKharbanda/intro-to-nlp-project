import csv
from collections import defaultdict

TEST_CSV = "test.csv"
TRAIN_DIR = "data"

# detect language using script detection
import unicodedata

def detect_language(text):
    for ch in text:
        name = unicodedata.name(ch, "")
        if "HIRAGANA" in name or "KATAKANA" in name:
            return "ja"
        if "CJK UNIFIED" in name:
            return "zh"
        if "HANGUL" in name:
            return "ko"
        if "ARABIC" in name:
            return "ar"
        if "DEVANAGARI" in name:
            return "hi"
        if "CYRILLIC" in name:
            return "ru"
    return "en"

with open(TEST_CSV, newline='', encoding="utf8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row["context"]
        lang = detect_language(text)
        with open(f"{TRAIN_DIR}/{lang}.txt", "a", encoding="utf8") as out:
            out.write(text + "\n")

print("Appended test.csv contexts to training data.")