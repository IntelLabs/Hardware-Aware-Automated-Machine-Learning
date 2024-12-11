import re
from datasets import Dataset, load_dataset

def add_prompt_func_piqa(document):
    """Add prompt to PIQA dataset document.

    Args:
        document (dict): A dictionary containing the PIQA dataset document.

    Returns:
        dict: The document with the added prompt.
    """
    def _process_document(document):
        """Process the PIQA document to extract relevant fields."""
        processed_document = {
            "goal": document["goal"],
            "choices": [document["sol1"], document["sol2"]],
            "gold": document["label"],
        }
        return processed_document

    def document_to_text(document):
        """Convert the PIQA document to text format."""
        return "Question: " + document["goal"] + "\nAnswer:"

    document = _process_document(document)
    prompt = document_to_text(document)
    answer = document["choices"][document["gold"]]
    document["full_prompt"] = prompt + " " + answer
    return document

def add_prompt_func_arc(document):
    """Add prompt to ARC dataset document.

    Args:
        document (dict): A dictionary containing the ARC dataset document.

    Returns:
        dict: The document with the added prompt.
    """
    def _process_document(document):
        """Process the ARC document to extract relevant fields."""
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        document["answerKey"] = num_to_letter.get(document["answerKey"], document["answerKey"])
        processed_document = {
            "id": document["id"],
            "query": "Question: " + document["question"] + "\nAnswer:",
            "choices": document["choices"]["text"],
            "gold": ["A", "B", "C", "D", "E"].index(document["answerKey"]),
        }
        return processed_document

    def document_to_text(document):
        """Convert the ARC document to text format."""
        return document["query"]

    document = _process_document(document)
    prompt = document_to_text(document)
    answer = document["choices"][document["gold"]]
    document["full_prompt"] = prompt + " " + answer
    return document

def add_prompt_func_hellaswag(document):
    """Add prompt to HellaSwag dataset document.

    Args:
        document (dict): A dictionary containing the HellaSwag dataset document.

    Returns:
        dict: The document with the added prompt.
    """
    def preprocess(text):
        """Preprocess the text by removing unwanted characters and formatting."""
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def _process_document(document):
        """Process the HellaSwag document to extract relevant fields."""
        context = document["ctx_a"] + " " + document["ctx_b"].capitalize()
        processed_document = {
            "query": preprocess(document["activity_label"] + ": " + context),
            "choices": [preprocess(ending) for ending in document["endings"]],
            "gold": int(document["label"]),
        }
        return processed_document

    def document_to_text(document):
        """Convert the HellaSwag document to text format."""
        return document["query"]

    document = _process_document(document)
    prompt = document_to_text(document)
    answer = document["choices"][document["gold"]]
    document["full_prompt"] = prompt + " " + answer
    return document

def add_prompt_func_openbookqa(document):
    """Add prompt to OpenBookQA dataset document.

    Args:
        document (dict): A dictionary containing the OpenBookQA dataset document.

    Returns:
        dict: The document with the added prompt.
    """
    def _process_document(document):
        """Process the OpenBookQA document to extract relevant fields."""
        processed_document = {
            "id": document["id"],
            "query": document["question_stem"],
            "choices": document["choices"]["text"],
            "gold": ["A", "B", "C", "D"].index(document["answerKey"].strip()),
        }
        return processed_document

    def document_to_text(document):
        """Convert the OpenBookQA document to text format."""
        return document["query"]

    document = _process_document(document)
    prompt = document_to_text(document)
    answer = document["choices"][document["gold"]]
    document["full_prompt"] = prompt + " " + answer
    return document

def add_prompt_func_boolq(document):
    """Add prompt to BoolQ dataset document.

    Args:
        document (dict): A dictionary containing the BoolQ dataset document.

    Returns:
        dict: The document with the added prompt.
    """
    def document_to_text(document):
        """Convert the BoolQ document to text format."""
        return f"{document['passage']}\nQuestion: {document['question']}?\nAnswer:"

    prompt = document_to_text(document)
    answer = "yes" if document['answer'] else "no"
    document["full_prompt"] = prompt + " " + answer
    return document

def add_prompt_func_winogrande(document):
    """Add prompt to WinoGrande dataset document.

    Args:
        document (dict): A dictionary containing the WinoGrande dataset document.

    Returns:
        dict: The document with the added prompt.
    """
    def document_to_text(document):
        """Convert the WinoGrande document to text format."""
        pronoun_location = document["sentence"].index("_")
        answer = document["option" + document["answer"]]
        return document["sentence"][:pronoun_location] + answer + document["sentence"][pronoun_location+1:]

    full_prompt = document_to_text(document)
    document["full_prompt"] = full_prompt
    return document

def load_cs_dataset(split="train", debug=False):
    """Load and combine multiple datasets for commonsense reasoning tasks.

    Args:
        split (str): The dataset split to load (default is "train").
        debug (bool): Whether to load only a subset of the data for debugging (default is False).

    Returns:
        Dataset: The combined dataset with prompts.
    """
    if debug:
        arc_challenge_dataset = load_dataset("ai2_arc", "ARC-Challenge", split=split)
        arc_challenge_dataset = arc_challenge_dataset.map(add_prompt_func_arc)
        combined_dataset_dict = {"full_prompt": arc_challenge_dataset["full_prompt"]}
        combined_dataset = Dataset.from_dict(combined_dataset_dict)
        return combined_dataset

    winogrande_dataset = load_dataset("winogrande", "winogrande_debiased", split=split)
    winogrande_dataset = winogrande_dataset.map(add_prompt_func_winogrande)

    boolq_dataset = load_dataset("boolq", split=split)
    boolq_dataset = boolq_dataset.map(add_prompt_func_boolq)

    openbookqa_dataset = load_dataset("openbookqa", split=split)
    openbookqa_dataset = openbookqa_dataset.map(add_prompt_func_openbookqa)

    hellaswag_dataset = load_dataset("hellaswag", split=split)
    hellaswag_dataset = hellaswag_dataset.map(add_prompt_func_hellaswag)

    piqa_dataset = load_dataset("piqa", split=split)
    piqa_dataset = piqa_dataset.map(add_prompt_func_piqa)

    arc_challenge_dataset = load_dataset("ai2_arc", "ARC-Challenge", split=split)
    arc_challenge_dataset = arc_challenge_dataset.map(add_prompt_func_arc)

    arc_easy_dataset = load_dataset("ai2_arc", "ARC-Easy", split=split)
    arc_easy_dataset = arc_easy_dataset.map(add_prompt_func_arc)

    combined_dataset = (openbookqa_dataset["full_prompt"] + hellaswag_dataset["full_prompt"] +
                        piqa_dataset["full_prompt"] + arc_challenge_dataset["full_prompt"] + arc_easy_dataset["full_prompt"] +
                        boolq_dataset["full_prompt"] + winogrande_dataset["full_prompt"])

    combined_dataset_dict = {"full_prompt": combined_dataset}
    combined_dataset = Dataset.from_dict(combined_dataset_dict)
    return combined_dataset

def add_prompt_func_gsm8k(document):
    """Add prompt to GSM8K dataset document.

    Args:
        document (dict): A dictionary containing the GSM8K dataset document.

    Returns:
        dict: The document with the added prompt.
    """
    def document_to_text(document):
        return f"Question: {document['question']}\nAnswer:"

    prompt = document_to_text(document)
    answer = document['answer']
    document["full_prompt"] = prompt + " " + answer
    return document

def load_gsm8k_dataset(split="train", debug=False):
    """Load the GSM8K dataset and add prompts.

    Args:
        split (str): The dataset split to load (default is "train").
        debug (bool): Whether to load only a subset of the data for debugging (default is False).

    Returns:
        Dataset: The GSM8K dataset with prompts.
    """
    gsm8k_dataset = load_dataset("gsm8k", "main", split=split)
    gsm8k_dataset = gsm8k_dataset.map(add_prompt_func_gsm8k)

    dataset_dict = {"full_prompt": gsm8k_dataset["full_prompt"]}
    dataset = Dataset.from_dict(dataset_dict)
    return dataset
