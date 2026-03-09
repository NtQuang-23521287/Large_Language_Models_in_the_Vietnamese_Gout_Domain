from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class GenerationResult:
    text: str
    meta: Dict[str, Any]

class BaseAdapter(ABC):
    """Abstract interface for all model adapteres. """

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        """Generate a response from a prompt."""
        raise NotImplementedError