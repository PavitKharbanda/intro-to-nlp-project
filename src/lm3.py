print("RUNNING THIS LM FILE")

import os, re
import pickle
import argparse
import unicodedata
from collections import defaultdict, Counter
import math
import gzip

N = 10
MIN_COUNT = 2

LANG_N = {
    "ja": 7,
    "ko": 8,
    "zh": 8, #before was 6
    "ar": 10,
    "hi": 10,
    "ru": 10,
    "en": 12,
    "fr": 12,
    "de": 12,
    "it": 12,
}


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


class CharNGramModel:
    def __init__(self, n=N):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        # self.context_totals = Counter()
        self.global_counts = Counter()
        self.global_top = []
        self.vocab = set()

    def train_stream(self, text):
        text = clean_text(text)
        padded = " " * (self.n - 1) + text

        for ch in text:
            self.global_counts[ch] += 1
            self.vocab.add(ch)

        self.vocab.add(" ")

        for i in range(self.n - 1, len(padded)):
            for order in range(1, self.n + 1):
                if i - order < 0:
                    continue

                context = padded[i - order:i]
                next_char = padded[i]

                self.ngram_counts[context][next_char] += 1
                # self.context_totals[context] += 1

    def prune(self):
        to_delete = []

        for ctx, counter in self.ngram_counts.items():
            if sum(counter.values()) < MIN_COUNT:
                to_delete.append(ctx)

        for ctx in to_delete:
            del self.ngram_counts[ctx]

        self.global_top = [
            ch for ch, _ in self.global_counts.most_common(10)
        ]


    


    def predict(self, context):
      context = clean_text(context)
      combined = Counter()
      
      for order in range(1, self.n + 1):
          if order == 1:
              ctx = ""
          else:
              if len(context) < order - 1:
                  continue
              ctx = context[-(order - 1):]
          
          if ctx in self.ngram_counts:
              counter = self.ngram_counts[ctx]
              total = sum(counter.values())
              if total < 2:
                  continue
              weight = 2.0 ** (order - 1)  # exponentially prefer longer contexts
              for ch, cnt in counter.items():
                  combined[ch] += weight * cnt / total
      
      if combined:
          preds = [ch for ch, _ in combined.most_common(3)]
      else:
          preds = []
      
      for ch in self.global_top:
          if len(preds) == 3:
              break
          if ch not in preds:
              preds.append(ch)
      
      return preds[:3]

    def score(self, text):
        text = clean_text(text)
        padded = " " * (self.n - 1) + text

        score = 0.0
        vocab_size = len(self.vocab)

        for i in range(self.n - 1, len(padded)):

            total_prob = 0.0
            orders_used = 0

            for order in range(1, self.n + 1):

                if order == 1:
                    ctx = ""
                else:
                    if i - order < 0:
                        continue
                    ctx = padded[i - order:i]

                if ctx in self.ngram_counts:
                    counter = self.ngram_counts[ctx]
                    total = sum(counter.values())
                    count = counter.get(padded[i], 0)
                    prob = (count + 0.5) / (total + 0.5 * vocab_size)
                else:
                    prob = 1 / vocab_size

                total_prob += prob
                orders_used += 1

            # average probability across orders
            avg_prob = total_prob / orders_used
            score += math.log(avg_prob)

        return score / max(len(text), 1)
    
    def save(self, path):
      with gzip.open(path, "wb") as f:
          pickle.dump(
              (self.n,
              self.ngram_counts,
              self.global_counts,
              self.global_top,
              self.vocab),
              f,
              protocol=pickle.HIGHEST_PROTOCOL
          )

    def load(self, path):
      with gzip.open(path, "rb") as f:
          (self.n,
          self.ngram_counts,
          self.global_counts,
          self.global_top,
          self.vocab) = pickle.load(f)



def detect_latin_variant(text):
    t = text.lower()

    if any(ch in t for ch in "äöüß"):
        return "de"

    if any(ch in t for ch in "œç"):
        return "fr"

    # Italian accents are weak signals — remove this rule
    return None


def detect_language_by_model(text, models):

    best_lang = None
    best_score = float("-inf")

    for lang, model in models.items():
        s = model.score(text)

        if s > best_score:
            best_score = s
            best_lang = lang

    return best_lang

def detect_language(text):
    has_hiragana = False
    has_katakana = False
    has_cjk = False

    for ch in text:
        name = unicodedata.name(ch, "")

        if "HIRAGANA" in name:
            has_hiragana = True
        elif "KATAKANA" in name:
            has_katakana = True
        elif "CJK UNIFIED" in name:
            has_cjk = True
        elif "HANGUL" in name:
            return "ko"
        elif "ARABIC" in name:
            return "ar"
        elif "DEVANAGARI" in name:
            return "hi"
        elif "CYRILLIC" in name:
            return "ru"

    # If kana exists → definitely Japanese
    if has_hiragana or has_katakana:
        return "ja"

    # If only kanji → ambiguous between zh and ja
    # if has_cjk:
    #     return None
    if has_cjk and not (has_hiragana or has_katakana):
      return "zh"

    return "en"

def train_model(work_dir):
    os.makedirs(work_dir, exist_ok=True)

    for fname in os.listdir("data"):
        if not fname.endswith(".txt"):
            continue

        lang = fname.replace(".txt", "")
        n = LANG_N.get(lang, N)
        print(f"Training {lang} with N={n}...")

        model = CharNGramModel(n=n)

        with open(os.path.join("data", fname), "r", encoding="utf8") as f:
            model.train_stream(f.read())

        model.prune()
        model.save(os.path.join(work_dir, f"{lang}.checkpoint"))

    print("Training complete.")


def test_model(work_dir, test_data, test_lang, test_output):

    models = {}

    for file in os.listdir(work_dir):
        if file.endswith(".checkpoint"):
            lang = file.replace(".checkpoint", "")
            m = CharNGramModel()
            m.load(os.path.join(work_dir, file))
            models[lang] = m

    with open(test_data, "r", encoding="utf8") as f:
        contexts = f.read().splitlines()

    with open(test_lang, "r", encoding="utf8") as f:
        langs = f.read().splitlines()

    print("Contexts:", len(contexts))
    print("Langs:", len(langs))

    assert len(contexts) == len(langs), "Mismatch between input and lang file!"

    with open(test_output, "w", encoding="utf8") as out:
        for context, lang in zip(contexts, langs):

            if lang not in models:
                out.write("   \n")
                continue

            preds = models[lang].predict(context)
            out.write("".join(preds) + "\n")

    print("Testing complete.")


def test_kaggle(work_dir, test_csv, output_csv):
    import csv

    models = {}

    for file in os.listdir(work_dir):
        if file.endswith(".checkpoint"):
            lang = file.replace(".checkpoint", "")
            m = CharNGramModel()
            m.load(os.path.join(work_dir, file))
            models[lang] = m

    with open(test_csv, newline='', encoding="utf8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with open(output_csv, "w", newline='', encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prediction"])

        for row in rows:
            idx = row["id"]
            context = row["context"]

            script_lang = detect_language(context)

            if script_lang and script_lang != "en":
                lang = script_lang
            else:
                # Let model decide for ambiguous cases (including CJK-only)
                latin_guess = detect_latin_variant(context)
                if latin_guess:
                    lang = latin_guess
                else:
                    lang = detect_language_by_model(context, models)

            if lang not in models:
                lang = "en"   # safe fallback

            preds = models[lang].predict(context)
            writer.writerow([idx, "".join(preds)])

    print("Submission written:", output_csv)


def test_without_langfile(work_dir, test_data, true_lang_file, answer_file, output_pred=None):

    models = {}

    for file in os.listdir(work_dir):
        if file.endswith(".checkpoint"):
            lang = file.replace(".checkpoint", "")
            m = CharNGramModel()
            m.load(os.path.join(work_dir, file))
            models[lang] = m

    with open(test_data, "r", encoding="utf8") as f:
        contexts = f.read().splitlines()

    with open(true_lang_file, "r", encoding="utf8") as f:
        true_langs = f.read().splitlines()

    with open(answer_file, "r", encoding="utf8") as f:
        answers = f.read().splitlines()

    assert len(contexts) == len(true_langs) == len(answers)

    total = 0
    correct_lang = 0
    correct_char = 0

    # Open output file once at start
    out_file = None
    if output_pred:
        out_file = open(output_pred, "w", encoding="utf8")

    for i, (context, true_lang, true_char) in enumerate(zip(contexts, true_langs, answers)):

        script_lang = detect_language(context)

        script_lang = detect_language(context)

        if script_lang and script_lang != "en":
            predicted_lang = script_lang
        else:
            # Let model decide for ambiguous cases (including CJK-only)
            latin_guess = detect_latin_variant(context)
            if latin_guess:
                predicted_lang = latin_guess
            else:
                predicted_lang = detect_language_by_model(context, models)


        if predicted_lang == true_lang:
            correct_lang += 1

        if predicted_lang not in models:
            predicted_lang = "en"

        preds = models[predicted_lang].predict(context)

        if out_file:
            out_file.write("".join(preds) + "\n")

            # Optional: flush every 1000 lines so you see progress
            if i % 1000 == 0:
                out_file.flush()
                print(f"Processed {i}/{len(contexts)} ({i/len(contexts)*100:.2f}%)")

        if true_char in preds:
            correct_char += 1

        total += 1

    if out_file:
        out_file.close()

    print("Language detection accuracy:", correct_lang / total)
    print("Character prediction accuracy:", correct_char / total)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test", "kaggle", "eval_no_lang"]) 
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--test_data")
    parser.add_argument("--test_lang")
    parser.add_argument("--test_output")
    parser.add_argument("--test_csv")
    parser.add_argument("--output_csv")  
    parser.add_argument("--true_lang")
    parser.add_argument("--answer")
    parser.add_argument("--output_pred")
    args = parser.parse_args()

    if args.mode == "train":
        train_model(args.work_dir)

    elif args.mode == "test":
        test_model(args.work_dir,
                   args.test_data,
                   args.test_lang,
                   args.test_output)
    
    elif args.mode == "kaggle":
        test_kaggle(args.work_dir,
                    args.test_csv,
                    args.output_csv)
        
    elif args.mode == "eval_no_lang":
        test_without_langfile(
            args.work_dir,
            args.test_data,
            args.true_lang,
            args.answer,
            args.output_pred
        )