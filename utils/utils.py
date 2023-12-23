def set_seed(seed: int = 19980406):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def normalize_document(document: str):
    document = document.replace("\n", " ").replace("’", "'")
    if document.startswith('"'):
        document = document[1:]
    if document.endswith('"'):
        document = document[:-1]
    return document

def normalize_query(question: str) -> str:
    question = question.replace("’", "'")
    return question

def get_yaml_file(file_path):
    import yaml  
    with open(file_path, "r") as file:  
        config = yaml.safe_load(file)  
    return config  


def get_linear_scheduler(
    optimizer,
    warmup_steps,
    total_training_steps,
    steps_shift=0,
    last_epoch=-1,
):
    from torch.optim.lr_scheduler import LambdaLR
    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        current_step += steps_shift
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            1e-7,
            float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)