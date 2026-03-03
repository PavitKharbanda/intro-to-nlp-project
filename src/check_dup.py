# file_path = "data/en.txt"

# with open(file_path, "r", encoding="utf8") as f:
#     lines = [line.rstrip("\n") for line in f]

# start1 = 245340
# start2 = 286608

# max_len = 0

# while (start1 + max_len < len(lines) and
#        start2 + max_len < len(lines) and
#        lines[start1 + max_len] == lines[start2 + max_len]):
#     max_len += 1

# print("Duplicate block length:", max_len)
# print("Block 1:", start1, "to", start1 + max_len - 1)
# print("Block 2:", start2, "to", start2 + max_len - 1)

#237361 - 250064 en
#278629 - 291332 en

#242154 - 255092 fr
#284204 - 297142 fr

#240002 - 252942 it
#282052 - 294990 it

#241977 - 254915 ru
#284027 - 296965 ru

#242254 - 255128 ko
#284238 - 297176 ko

#239250 - 252105 zh
#281215 - 294153 zh

#242458 - 255398 ar
#284508 - 297446 ar

#242380 - 255320 de
#281596 - 294534 de

#242233 - 255171 hi
#284283 - 297221 hi

#242380 - 253837 ja
#284430 - 295884 ja



#_pagina: 148 _nastro: 3, 2, 1, 0 Roger.
# Stony 10, 9, 8, 7, 6, 5, 4, 3, CC Decollo.


# file_path = "data/en.txt"

# with open(file_path, "r", encoding="utf8") as f:
#     lines = [line.rstrip("\n") for line in f]

# block_start = 278629
# block_end = 291332
# block = lines[block_start:block_end+1]

# # Search earlier for matching block
# for i in range(block_start):
#     if lines[i:i+len(block)] == block:
#         print("Original block found at:", i, "to", i+len(block)-1)
#         break


import os

# Define ranges (0-based indexing assumed below)
duplicate_ranges = {
    "en": [(237361, 250064), (278629, 291332)],
    "fr": [(242154, 255092), (284204, 297142)],
    "it": [(240002, 252942), (282052, 294990)],
    "ru": [(241977, 254915), (284027, 296965)],
    "ko": [(242254, 255128), (284238, 297176)],
    "zh": [(239250, 252105), (281215, 294153)],
    "ar": [(242458, 255398), (284508, 297446)],
    "de": [(242380, 255320), (281596, 294534)],
    "hi": [(242233, 255171), (284283, 297221)],
    "ja": [(242380, 253837), (284430, 295884)],
}

DATA_DIR = "data"

def verify_blocks(lang, ranges):
    file_path = os.path.join(DATA_DIR, f"{lang}.txt")

    with open(file_path, "r", encoding="utf8") as f:
        lines = [line.rstrip("\n") for line in f]

    (s1, e1), (s2, e2) = ranges

    block1 = lines[s1:e1+1]
    block2 = lines[s2:e2+1]

    print(f"\n=== {lang.upper()} ===")
    print("Block1 length:", len(block1))
    print("Block2 length:", len(block2))

    if len(block1) != len(block2):
        print("❌ Length mismatch!")
        return

    if block1 == block2:
        print("✅ Blocks are EXACTLY identical.")
    else:
        print("❌ Blocks differ. Checking first mismatch...")
        for i, (l1, l2) in enumerate(zip(block1, block2)):
            if l1 != l2:
                print("Mismatch at relative line:", i)
                print("Block1:", l1)
                print("Block2:", l2)
                break


for lang, ranges in duplicate_ranges.items():
    verify_blocks(lang, ranges)