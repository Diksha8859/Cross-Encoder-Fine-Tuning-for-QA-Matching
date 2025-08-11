def build_lookup_table(dataset):
    """
    Build a lookup table with normalized utterances as keys and labels as values.
    """
    lookup = {}
    for sample in dataset:
        norm_text = normalize_text(sample["input_text"])
        lookup[norm_text] = sample["output_label"]
    return lookup

def find_label(utterance, lookup_table):
    """
    Find the label for a given utterance using the lookup table.
    """
    normalized_utterance = normalize_text(utterance)
    return lookup_table.get(normalized_utterance, "Label not found")
