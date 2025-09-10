
def doc_to_text(doc):
    return f"Question: {doc['question']}\n Options:\n{doc['options']}.\nPlease only output the choice letter in the answer field e.g. Final Answer: A"

def doc_to_target(doc):
    if doc['answer'] == 'A':
        return 0
    elif doc['answer'] == 'B':
        return 1
    elif doc['answer'] == 'C':
        return 2
    elif doc['answer'] == 'D':
        return 3
