# #!/usr/bin/env python
# import os
# import string
# import random
# from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


# class MyModel:
#     """
#     This is a starter model to get you started. Feel free to modify this file.
#     """

#     @classmethod
#     def load_training_data(cls):
#         # your code here
#         # this particular model doesn't train
#         return []

#     @classmethod
#     def load_test_data(cls, fname):
#         # your code here
#         data = []
#         with open(fname) as f:
#             for line in f:
#                 inp = line[:-1]  # the last character is a newline
#                 data.append(inp)
#         return data

#     @classmethod
#     def write_pred(cls, preds, fname):
#         with open(fname, 'wt') as f:
#             for p in preds:
#                 f.write('{}\n'.format(p))

#     def run_train(self, data, work_dir):
#         # your code here
#         pass

#     def run_pred(self, data):
#         # your code here
#         preds = []
#         all_chars = string.ascii_letters
#         for inp in data:
#             # this model just predicts a random character each time
#             top_guesses = [random.choice(all_chars) for _ in range(3)]
#             preds.append(''.join(top_guesses))
#         return preds

#     def save(self, work_dir):
#         # your code here
#         # this particular model has nothing to save, but for demonstration purposes we will save a blank file
#         with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
#             f.write('dummy save')

#     @classmethod
#     def load(cls, work_dir):
#         # your code here
#         # this particular model has nothing to load, but for demonstration purposes we will load a blank file
#         with open(os.path.join(work_dir, 'model.checkpoint')) as f:
#             dummy_save = f.read()
#         return MyModel()


# if __name__ == '__main__':
#     parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
#     parser.add_argument('mode', choices=('train', 'test'), help='what to run')
#     parser.add_argument('--work_dir', help='where to save', default='work')
#     parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
#     parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
#     args = parser.parse_args()

#     random.seed(0)

#     if args.mode == 'train':
#         if not os.path.isdir(args.work_dir):
#             print('Making working directory {}'.format(args.work_dir))
#             os.makedirs(args.work_dir)
#         print('Instatiating model')
#         model = MyModel()
#         print('Loading training data')
#         train_data = MyModel.load_training_data()
#         print('Training')
#         model.run_train(train_data, args.work_dir)
#         print('Saving model')
#         model.save(args.work_dir)
#     elif args.mode == 'test':
#         print('Loading model')
#         model = MyModel.load(args.work_dir)
#         print('Loading test data from {}'.format(args.test_data))
#         test_data = MyModel.load_test_data(args.test_data)
#         print('Making predictions')
#         pred = model.run_pred(test_data)
#         print('Writing predictions to {}'.format(args.test_output))
#         assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
#         model.write_pred(pred, args.test_output)
#     else:
#         raise NotImplementedError('Unknown mode {}'.format(args.mode))


# import os, re, unicodedata
# import math
# import pickle
# import argparse
# from collections import defaultdict, Counter

# # ================= CONFIG =================
# N = 5
# K = 0.5

# # ================= CLEANING =================
# def clean_text(text):
#     text = text.lower()
    
#     # Normalize unicode (very important)
#     text = unicodedata.normalize("NFKC", text)
    
#     # Remove control characters
#     text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("C"))
    
#     # Normalize whitespace
#     text = re.sub(r"\s+", " ", text)
    
#     return text.strip()

# # ================= MODEL =================
# class CharNGramModel:
#     def __init__(self, n=N):
#         self.n = n
#         self.counts = [defaultdict(Counter) for _ in range(n)]
#         self.vocab = set()

#     def train_line(self, line):
#         line = clean_text(line.rstrip("\n"))
#         padded = " " * (self.n - 1) + line

#         for i in range(self.n - 1, len(padded)):
#             for order in range(1, self.n + 1):
#                 if i - order < 0:
#                     continue
#                 context = padded[i - order:i]
#                 next_char = padded[i]
#                 self.counts[order - 1][context][next_char] += 1
#                 self.vocab.add(next_char)

#     def train(self, lines):
#         for line in lines:
#             self.train_line(line)

#     # ================= LANGUAGE SCORING =================
#     def score(self, context):
#         context = clean_text(context)
#         padded = " " * (self.n - 1) + context
#         log_prob = 0.0

#         for i in range(self.n - 1, len(padded)):
#             found = False
#             for order in range(self.n, 0, -1):
#                 if i - order < 0:
#                     continue
#                 ctx = padded[i - order:i]
#                 if ctx in self.counts[order - 1]:
#                     counter = self.counts[order - 1][ctx]
#                     total = sum(counter.values())
#                     vocab_size = len(self.vocab)
#                     ch = padded[i]
#                     prob = (counter[ch] + K) / (total + K * vocab_size)
#                     log_prob += math.log(prob)
#                     found = True
#                     break
#             if not found:
#                 log_prob += math.log(1e-8)

#         return log_prob

#     # ================= NEXT CHAR PREDICTION =================
#     def predict(self, context):
#         context = clean_text(context)
#         context = context[-(self.n - 1):]

#         for order in range(self.n, 0, -1):
#             if len(context) < order - 1:
#                 continue

#             ctx = context[-(order - 1):]
#             if ctx in self.counts[order - 1]:
#                 counter = self.counts[order - 1][ctx]
#                 total = sum(counter.values())
#                 vocab_size = len(self.vocab)

#                 probs = []
#                 for ch in self.vocab:
#                     prob = (counter[ch] + K) / (total + K * vocab_size)
#                     probs.append((prob, ch))

#                 probs.sort(reverse=True)
#                 return [ch for _, ch in probs[:3]]

#         return [' ', 'e', 't']


# # ================= TRAIN =================
# def train_model(work_dir):
#     models = {}

#     data_path = os.path.join(os.getcwd(), "data")

#     for fname in os.listdir(data_path):
#         if not fname.endswith(".txt"):
#             continue

#         lang = fname.replace(".txt", "")
#         model = CharNGramModel()

#         file_path = os.path.join(data_path, fname)
#         with open(file_path, "r", encoding="utf8") as f:
#             lines = f.readlines()
#             model.train(lines)

#         models[lang] = model
#         print(f"Trained LM for {lang}")

#     os.makedirs(work_dir, exist_ok=True)

#     with open(os.path.join(work_dir, "all_models.pkl"), "wb") as f:
#         pickle.dump(models, f)

#     print("Training complete.")


# # ================= TEST =================
# def test_model(work_dir, test_data, test_output):
#     with open(os.path.join(work_dir, "all_models.pkl"), "rb") as f:
#         models = pickle.load(f)

#     with open(test_data, "r", encoding="utf8") as f:
#         lines = f.readlines()

#     with open(test_output, "w", encoding="utf8") as out:
#         for line in lines:
#             line = line.rstrip("\n")

#             best_lang = None
#             best_score = -1e18

#             for lang, model in models.items():
#                 score = model.score(line)
#                 if score > best_score:
#                     best_score = score
#                     best_lang = lang

#             preds = models[best_lang].predict(line)
#             out.write("".join(preds) + "\n")

#     print("Prediction complete.")


# # ================= MAIN =================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("mode", choices=["train", "test"])
#     parser.add_argument("--work_dir", required=True)
#     parser.add_argument("--test_data")
#     parser.add_argument("--test_output")

#     args = parser.parse_args()

#     if args.mode == "train":
#         train_model(args.work_dir)

#     elif args.mode == "test":
#         test_model(args.work_dir, args.test_data, args.test_output)


import os
import re
import pickle
import argparse
import unicodedata
from collections import defaultdict, Counter

# =========================
# CONFIG
# =========================
N = 5

# Interpolation weights (tuned for leaderboard)
LAMBDAS = [0.05, 0.1, 0.15, 0.25, 0.45]  
# unigram → 5gram

# =========================
# CLEANING
# =========================
def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove metadata markers
    text = re.sub(r"_\w+", "", text)

    # Remove control characters
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("C"))

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================
# MODEL
# =========================
class CharNGramModel:
    def __init__(self, n=N):
        self.n = n
        self.counts = [defaultdict(Counter) for _ in range(n)]
        self.vocab = set()

    def train_line(self, line):
        line = clean_text(line)
        if not line:
            return

        padded = " " * (self.n - 1) + line

        for ch in line:
            self.vocab.add(ch)
        self.vocab.add(" ")

        for i in range(self.n - 1, len(padded)):
            for order in range(1, self.n + 1):
                if i - order < 0:
                    continue
                context = padded[i - order:i]
                next_char = padded[i]
                self.counts[order - 1][context][next_char] += 1

    def train(self, lines):
        for line in lines:
            self.train_line(line)

    # -------------------------
    # Interpolated Prediction
    # -------------------------
    def predict(self, context):
        context = clean_text(context)
        context = context[-(self.n - 1):]

        vocab = list(self.vocab)
        V = len(vocab)

        # Backoff from highest order to lowest
        for order in range(self.n, 0, -1):

            if len(context) < order - 1:
                continue

            ctx = context[-(order - 1):]

            if ctx in self.counts[order - 1]:
                counter = self.counts[order - 1][ctx]
                total = sum(counter.values())

                scores = []
                for ch in vocab:
                    # Add-K smoothing
                    prob = (counter[ch] + 0.5) / (total + 0.5 * V)
                    scores.append((prob, ch))

                scores.sort(reverse=True)
                return [ch for _, ch in scores[:3]]

        # Fallback
        return [' ', 'e', 't']

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.counts, self.vocab), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.counts, self.vocab = pickle.load(f)


# =========================
# TRAIN ONE MODEL PER LANGUAGE
# =========================
def train_model(work_dir):
    os.makedirs(work_dir, exist_ok=True)

    data_path = "data"

    for fname in os.listdir(data_path):
        if not fname.endswith(".txt"):
            continue

        lang = fname.replace(".txt", "")
        print(f"Training {lang}...")

        model = CharNGramModel()

        with open(os.path.join(data_path, fname), "r", encoding="utf8") as f:
            model.train(f.readlines())

        model.save(os.path.join(work_dir, f"{lang}.checkpoint"))

    print("Training complete.")


# =========================
# TEST
# =========================
def test_model(work_dir, test_data, test_lang, test_output):

    # Load all models
    models = {}
    for file in os.listdir(work_dir):
        if file.endswith(".checkpoint"):
            lang = file.replace(".checkpoint", "")
            model = CharNGramModel()
            model.load(os.path.join(work_dir, file))
            models[lang] = model

    with open(test_data, "r", encoding="utf8") as f:
        inputs = f.readlines()

    with open(test_lang, "r", encoding="utf8") as f:
        langs = f.readlines()

    with open(test_output, "w", encoding="utf8") as out:
        for line, lang in zip(inputs, langs):
            line = line.strip()
            lang = lang.strip()

            if lang not in models:
                out.write("   \n")
                continue

            preds = models[lang].predict(line)
            out.write("".join(preds) + "\n")

    print("Testing complete.")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--test_data")
    parser.add_argument("--test_lang")
    parser.add_argument("--test_output")

    args = parser.parse_args()

    if args.mode == "train":
        train_model(args.work_dir)

    elif args.mode == "test":
        test_model(args.work_dir, args.test_data, args.test_lang, args.test_output)