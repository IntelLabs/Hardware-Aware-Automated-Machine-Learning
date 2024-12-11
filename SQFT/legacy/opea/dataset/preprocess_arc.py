import json
from datasets import load_dataset

def process_arc_document(document):
    """Add prompt to ARC dataset document.

    Args:
        document (dict): A dictionary containing the ARC dataset document.

    Returns:
        dict: The document with the added prompt.
    """
    
    instruction = document["question"]
    answer_key = document["answerKey"]
    choices = document["choices"]["label"]
    texts = document["choices"]["text"]
    
    def _process_output(document):
        """Process the ARC document to extract relevant fields."""
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        document["answerKey"] = num_to_letter.get(document["answerKey"], document["answerKey"])
        processed_output = {
            "choices": document["choices"]["text"],
            "gold": ["A", "B", "C", "D", "E"].index(document["answerKey"]),
        }
        return processed_output
    
    
    processed_output = _process_output(document)
    answer = processed_output["choices"][processed_output["gold"]]
    new_entry = {
        "instruction": instruction,
        "input": "",
        "output": answer
    }
    return new_entry


dataset = load_dataset("ai2_arc", "ARC-Easy", split="train")
new_data = [process_arc_document(doc) for doc in dataset]

print(len(new_data))

with open("arce_train_instruct.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
