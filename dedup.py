# Open the original files for reading
from tqdm import tqdm
with open('train.en', 'r') as en_file, open('train.hi', 'r') as hi_file:
    # Read the contents of the files into lists
    en_lines = en_file.readlines()
    hi_lines = hi_file.readlines()
print("Files Read")

# Create a set to store the combined keys
keys_set = set()

# Iterate over the lines and add the combined keys to the set
for en_line, hi_line in zip(tqdm(en_lines), hi_lines):
    key = (en_line.strip(), hi_line.strip())
    keys_set.add(key)


# assert len(en_set) == len(hi_set)
# Open the output files for writing
with open('train.dedup.en', 'w') as dedup_en_file, open('train.dedup.hi', 'w') as dedup_hi_file:
    # Write the unique lines to the output files
    for en_line, hi_line in tqdm(list(keys_set), total=len(keys_set)):
        dedup_en_file.write(en_line+"\n")
        dedup_hi_file.write(hi_line+"\n")
