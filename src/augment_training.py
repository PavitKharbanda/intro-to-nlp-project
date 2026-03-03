import os

INPUT_FILE = "data/open-dev/input.txt"
LANG_FILE = "data/open-dev/lang.txt"
ANS_FILE = "data/open-dev/answers.txt"
TRAIN_DIR = "data"

def main():
    with open(INPUT_FILE, "r", encoding="utf8") as f:
        lines = f.read().splitlines()

    with open(ANS_FILE, "r", encoding="utf8") as f:
        answers = f.read().splitlines()

    with open(LANG_FILE, "r", encoding="utf8") as f:
        langs = f.read().splitlines()

    assert len(lines) == len(answers) == len(langs), "Mismatch between input, answers, and lang files"

    for line, ans, lang in zip(lines, answers, langs):
        train_file = os.path.join(TRAIN_DIR, f"{lang}.txt")

        if not os.path.exists(train_file):
            print(f"Skipping unknown language {lang}")
            continue

        with open(train_file, "a", encoding="utf8") as out:
            out.write(line + ans + "\n")

    print("Open-dev data appended to training files.")

if __name__ == "__main__":
    main()