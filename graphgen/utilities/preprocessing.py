"""Module containing utilities for preprocessing data."""
import re
from typing import Any, Dict


class QuestionPreprocessor:
    """Class for preprocessing questions."""

    KEY_MASK = ("imageId", "question", "answer")
    VOCAB_MASK = ("question", "answer")

    def __init__(self) -> None:
        """Create a `QuestionPreprocessor` instance."""
        self.word_to_index: Dict[str, int] = {}

    def __call__(self, question: Any) -> Any:
        """Preprocess a question sample."""
        # Filter out unused fields
        result = {key: val for key, val in question.items() if key in self.KEY_MASK}

        # Populate word_to_index dict
        for key, val in result.items():
            if key in self.VOCAB_MASK:
                lval = val.lower()
                lval = re.sub(r"[^\w\s]", "", lval)
                tokens = []
                for word in lval.split():
                    if word not in self.word_to_index.keys():
                        self.word_to_index[word] = len(self.word_to_index)
                    tokens.append(self.word_to_index[word])
                result[key] = tokens

        return result
