from transformers import BertConfig

class ColBERTConfig(BertConfig):
    def __init__(
        self,
        dim = 128,
        mask_punctuation = True,
        similarity_metric = 'l2', # [l2,cosine],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.mask_punctuation = mask_punctuation
        self.similarity_metric = similarity_metric