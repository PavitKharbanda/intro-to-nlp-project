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

#278629 - 291332

file_path = "data/en.txt"

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


def find_large_duplicate_block(file_path, window=500):
    with open(file_path, "r", encoding="utf8") as f:
        lines = [line.rstrip("\n") for line in f]

    n = len(lines)

    for i in range(n - window):
        chunk = tuple(lines[i:i+window])
        for j in range(i + window, n - window):
            if tuple(lines[j:j+window]) == chunk:
                print(f"Duplicate block detected in {file_path}")
                print(f"Block 1 starts at {i}")
                print(f"Block 2 starts at {j}")

                # now measure full block length
                length = 0
                while (
                    i + length < n and
                    j + length < n and
                    lines[i + length] == lines[j + length]
                ):
                    length += 1

                print(f"Full duplicate length: {length}")
                print(f"Block 1: {i} to {i+length-1}")
                print(f"Block 2: {j} to {j+length-1}")
                return (i, j, length)

    print(f"No large duplicate block found in {file_path}")
    return None

find_large_duplicate_block(file_path)