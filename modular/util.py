





def prompt_template(model_n, tokenizer):

    if model_n == "llama2_7b" or model_n == "llama2_13b":
        return  """<s>[INST] <<SYS>>\n Give a short answer to the question .\n<</SYS>>\n\nQuestion: {question} [/INST]""", """<s>[INST] <<SYS>>\nGive a short answer to the question based on the context.\n<</SYS>>\n\nContext:
                {context}\n\n Question: {question} [/INST]"""
    else:
        prompt_in_chat_format_rag = [
        {
            "role": "system",
            "content": """
        Give a short answer to the question based on the context.
        """,
            },
            {
                "role": "user",
                "content": """Context:
        {context}
        ---
        Question: {question}""",
            },         
        ]
        
        prompt_in_chat_format_llm = [
            {
                "role": "system",
                "content": """
         Give n short answer to the question""",},
            {
                "role": "user",
                "content": """
        Question: {question}""",
            },         
        ]

        PROMPT_TEMPLATE_RAG = tokenizer.apply_chat_template(
            prompt_in_chat_format_rag, tokenize=False, add_generation_prompt=True
        )
    
        PROMPT_TEMPLATE_LLM = tokenizer.apply_chat_template(
            prompt_in_chat_format_llm, tokenize=False, add_generation_prompt=True
        )
        
        
        
        return PROMPT_TEMPLATE_LLM, PROMPT_TEMPLATE_RAG


def eval_EM(answer, gts):
    for gt in gts:
        if gt in answer:
            return True
    return False

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

