from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from architecture.concept_modules import ConceptOutput

from . import tensor_to_numpy

__all__ = ["LinearClassifier"]


@dataclass
class ClassificationOutputNumpy:
    labels_logits: np.ndarray


@dataclass
class ClassificationOutput:
    labels_logits: torch.Tensor

    def to_numpy(self) -> ClassificationOutputNumpy:
        return ClassificationOutputNumpy(
            labels_logits=tensor_to_numpy(self.labels_logits),
        )


    
class ClassifcationModule(nn.Module):
    ...

def get_classification_module_by_name(name: str, **kwargs) -> ClassifcationModule:
    modules = {"LinearClassifier": LinearClassifier}

    if name not in modules:
        raise ValueError(f"Unknown Concept-Modul: {name}")
    return modules[name](**kwargs)


class LinearClassifier(ClassifcationModule):
    """Simple linear classification layer"""

    def __init__(self, n_concepts: int, n_classes: int, **kwargs):
        super().__init__()
        self.fc = nn.Linear(n_concepts, n_classes)

    def forward(self, concept_output: ConceptOutput) -> ClassificationOutput:
        labels_logits = self.fc(concept_output.concept_logits)
        
        return ClassificationOutput(labels_logits=labels_logits)
