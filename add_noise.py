import random
from tqdm import tqdm
random.seed(42)
def add_noise_natural(lines, noise_level=0.01):
    """
    Adds natural noise over a list of lines.

    Args:
        lines (list of str): The lines to add noise to.
        noise_level (float): The level of noise to add, as a fraction of the line length.

    Returns:
        A list of lines with natural noise added.
    """
    # Create a list to store the noisy lines
    noisy_lines = []

    # Iterate over the lines and add noise to each one
    for line in tqdm(lines):
        # Determine the amount of noise to add based on the line length and noise level
        noise_amount = int(len(line) * noise_level)

        # Generate a list of indices to add noise to
        indices = random.sample(range(len(line)), noise_amount)

        # Replace the characters at the noisy indices with random characters
        noisy_line = list(line)
        for i in indices:
            noisy_line[i] = chr(random.randint(ord('a'), ord('z')))
        noisy_line = ''.join(noisy_line)

        # Add the noisy line to the list of noisy lines
        noisy_lines.append(noisy_line)

    return noisy_lines

def add_keyboard_noise(lines, noise_level=0.05):
    """
    Adds keyboard noise over a list of lines.

    Args:
        lines (list of str): The lines to add noise to.
        noise_level (float): The level of noise to add, as a fraction of the line length.

    Returns:
        A list of lines with keyboard noise added.
    """
    # Define the keyboard layout
    keyboard = [
        [None, 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\'],
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\'', None, None],
        [None, 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', None, None]
    ]

    # Create a list to store the noisy lines
    noisy_lines = []

    # Iterate over the lines and add noise to each one
    for line in tqdm(lines):
        # Determine the amount of noise to add based on the line length and noise level
        noise_amount = int(len(line) * noise_level)

        # Generate a list of indices to add noise to
        indices = random.sample(range(len(line)), noise_amount)

        # Replace the characters at the noisy indices with nearby keyboard characters
        noisy_line = list(line)
        for i in indices:
            char = line[i]
            row = None
            col = None

            # Find the row and column of the character in the keyboard layout
            for r, row_chars in enumerate(keyboard):
                if char in row_chars:
                    row = r
                    col = row_chars.index(char)
                    break

            if row is not None and col is not None:
                # Find nearby keys in the keyboard layout
                nearby_rows = keyboard[max(row-1, 0):min(row+2, len(keyboard))]
                nearby_chars = []
                for r in nearby_rows:
                    if r is not None:
                        nearby_chars.extend(r[max(col-1, 0):min(col+2, len(r))])
                nearby_chars = [c for c in nearby_chars if c is not None and c != char]

                if nearby_chars:
                    # Replace the character with a nearby key with some probability
                    new_char = char
                    if random.random() < 0.5:
                        new_char = random.choice(nearby_chars)
                    noisy_line[i] = new_char

        noisy_line = ''.join(noisy_line)

        # Add the noisy line to the list of noisy lines
        noisy_lines.append(noisy_line)

    return noisy_lines

def add_vowel_noise(lines, noise_level=0.05):
    """
    Adds random removal of vowels as noise over a list of lines.

    Args:
        lines (list of str): The lines to add noise to.
        noise_level (float): The level of noise to add, as a fraction of the line length.

    Returns:
        A list of lines with vowel noise added.
    """
    # Define a list of vowels
    vowels = ['a', 'e', 'i', 'o', 'u']

    # Create a list to store the noisy lines
    noisy_lines = []

    # Iterate over the lines and add noise to each one
    for line in tqdm(lines):
        # Determine the amount of noise to add based on the line length and noise level
        noise_amount = int(len(line) * noise_level)

        # Generate a list of indices to add noise to
        indices = random.sample(range(len(line)), noise_amount)

        # Remove the vowels at the noisy indices
        noisy_line = list(line)
        for i in indices:
            if noisy_line[i].lower() in vowels:
                noisy_line[i] = ''

        noisy_line = ''.join(noisy_line)

        # Add the noisy line to the list of noisy lines
        noisy_lines.append(noisy_line)

    return noisy_lines

f = open('train.lc.en')
data = f.readlines()
for i in tqdm(range(len(data))):
    data[i] = data[i].strip()
data = add_noise_natural(data)
data = add_vowel_noise(data)
data = add_keyboard_noise(data)

with open('train.noise.en', 'w') as noise_en_file:
    # Write the unique lines to the output files
    for en_line in tqdm(data):
        noise_en_file.write(en_line+"\n")