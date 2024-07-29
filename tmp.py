import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
model = AutoModel.from_pretrained('facebook/contriever-msmarco')

sentences = [
    "Where was Marie Curie born?",
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
]

# Apply tokenizer
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
outputs = model(**inputs)

# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

print(outputs[0].shape)

embeddings = mean_pooling(outputs[0], inputs['attention_mask'])


score01 = embeddings[0] @ embeddings[1] #1.0473
score02 = embeddings[0] @ embeddings[2] #1.0095

print(score01)
print(score02)