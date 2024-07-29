from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from typing import Any, Dict, List, Optional
import sentence_transformers

class bge_embed_t(HuggingFaceBgeEmbeddings):

    def embed_documents(self, texts: List[str]) -> List[List[float]]:


    texts = [self.embed_instruction + t.replace("\n", " ") for t in texts]
    if self.multi_process:
        pool = self.client.start_multi_process_pool()
        embeddings = self.client.encode_multi_process(texts, pool)
        sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
    else:
        embeddings = self.client.encode(
            texts, show_progress_bar=self.show_progress, **self.encode_kwargs
        )
    return embeddings.tolist()

