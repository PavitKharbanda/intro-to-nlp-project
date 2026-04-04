# import os

# INPUT_FILE = "data/open-dev/input.txt"
# LANG_FILE = "data/open-dev/lang.txt"
# ANS_FILE = "data/open-dev/answers.txt"
# TRAIN_DIR = "data"

# def main():
#     with open(INPUT_FILE, "r", encoding="utf8") as f:
#         lines = f.read().splitlines()

#     with open(ANS_FILE, "r", encoding="utf8") as f:
#         answers = f.read().splitlines()

#     with open(LANG_FILE, "r", encoding="utf8") as f:
#         langs = f.read().splitlines()

#     assert len(lines) == len(answers) == len(langs), "Mismatch between input, answers, and lang files"

#     for line, ans, lang in zip(lines, answers, langs):
#         train_file = os.path.join(TRAIN_DIR, f"{lang}.txt")

#         if not os.path.exists(train_file):
#             print(f"Skipping unknown language {lang}")
#             continue

#         with open(train_file, "a", encoding="utf8") as out:
#             out.write(line + ans + "\n")

#     print("Open-dev data appended to training files.")

# if __name__ == "__main__":
#     main()

import os
from collections import defaultdict

INPUT_FILE = "data/open-dev/input.txt"
LANG_FILE = "data/open-dev/lang.txt"
ANS_FILE = "data/open-dev/answer.txt"
TRAIN_DIR = "data"

def main():

    with open(INPUT_FILE, "r", encoding="utf8") as f:
        inputs = f.read().splitlines()

    with open(ANS_FILE, "r", encoding="utf8") as f:
        answers = f.read().splitlines()

    with open(LANG_FILE, "r", encoding="utf8") as f:
        langs = f.read().splitlines()

    assert len(inputs) == len(answers) == len(langs)

    # Count how many lines were added per language
    lang_counts = defaultdict(int)
    for lang in langs:
        lang_counts[lang] += 1

    print("Open-dev counts per language:")
    for lang, count in lang_counts.items():
        print(lang, count)

    # STEP 1: Remove previously appended input-only lines
    for lang, count in lang_counts.items():
        train_file = os.path.join(TRAIN_DIR, f"{lang}.txt")

        if not os.path.exists(train_file):
            continue

        with open(train_file, "r", encoding="utf8") as f:
            lines = f.read().splitlines()

        print(f"Truncating {lang}: removing last {count} lines")

        lines = lines[:-count]

        with open(train_file, "w", encoding="utf8") as f:
            for line in lines:
                f.write(line + "\n")

    # STEP 2: Append corrected full lines
    for line, ans, lang in zip(inputs, answers, langs):
        train_file = os.path.join(TRAIN_DIR, f"{lang}.txt")

        if not os.path.exists(train_file):
            continue

        with open(train_file, "a", encoding="utf8") as f:
            f.write(line + ans + "\n")

    print("Rebuilt training data with full sentences.")

if __name__ == "__main__":
    main()