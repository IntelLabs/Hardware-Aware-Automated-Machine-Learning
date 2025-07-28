from docarray import BaseDoc
from docarray.typing import TorchTensor


class ClientRequestDoc(BaseDoc):
    query_embedding: TorchTensor
    num_retrieved_samples: int
