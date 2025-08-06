import torch

class LatentWorkspace:
    def __init__(self, dim=1024):
        self.workspace = torch.zeros((dim,))

    def update(self, embeddings: torch.Tensor):
        # TODO: merge new embeddings into workspace
        self.workspace = 0.9 * self.workspace + 0.1 * embeddings.mean(dim=0)

    def read(self) -> torch.Tensor:
        return self.workspace
