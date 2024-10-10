import os
import re

from utils.utils import crawl_directory

grToLat = {
    "Α": "A",
    "Ά": "A",
    "α": "a",
    "ά": "a",
    "Β": "B",
    "β": "b",
    "Γ": "G",
    "γ": "g",
    "Δ": "D",
    "δ": "d",
    "Ε": "E",
    "Έ": "E",
    "έ": "e",
    "ε": "e",
    "Ζ": "Z",
    "ζ": "z",
    "Η": "H",
    "Ή": "H",
    "η": "h",
    "ή": "h",
    "Θ": "U",
    "θ": "u",
    "Ι": "I",
    "Ί": "I",
    "ι": "i",
    "ί": "i",
    "Κ": "K",
    "κ": "k",
    "Λ": "L",
    "λ": "l",
    "Μ": "M",
    "μ": "m",
    "Ν": "N",
    "ν": "n",
    "Ξ": "J",
    "ξ": "j",
    "Ο": "O",
    "Ό": "O",
    "ο": "o",
    "ό": "o",
    "Π": "P",
    "π": "p",
    "Ρ": "R",
    "ρ": "r",
    "Σ": "S",
    "ς": "s",
    "σ": "s",
    "Τ": "T",
    "τ": "t",
    "Υ": "Y",
    "Ύ": "Y",
    "ύ": "y",
    "υ": "y",
    "Φ": "F",
    "φ": "f",
    "Χ": "X",
    "χ": "x",
    "Ψ": "C",
    "ψ": "c",
    "ω": "v",
    "ώ": "v",
    "Ω": "V",
    "Ώ": "V",
    " ": "_",
}


def remove_symbols(string):
    return re.sub(r"[^\w\s]", "", string)


def deEmojify(text):
    regrex_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        re.UNICODE,
    )
    return regrex_pattern.sub(r"", text)


def rename_content(path):
    tree = crawl_directory(path, ".wav")
    for filename in tree:
        removed = filename.maketrans(grToLat)
        dst = filename.translate(removed)
        dst = deEmojify(dst)
        name = os.path.splitext(dst)[0].split(os.sep)[-1]
        dst = dst.replace(name, remove_symbols(name))
        os.rename(filename, dst)
