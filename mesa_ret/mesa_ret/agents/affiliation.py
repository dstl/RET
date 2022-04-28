"""Agent affiliations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


class Affiliation(Enum):
    """An enum representing the different possible affiliations."""

    FRIENDLY = 1
    NEUTRAL = 2
    HOSTILE = 3
    UNKNOWN = 4

    def accept(self, visitor: AffiliationVisitor) -> None:
        """Accept a AffiliationVisitor and determines appropriate course of action.

        The purpose of the `else` clause is to future proof affiliation, in case new
        affiliations are added, but this method is not updated correspondingly.

        Args:
            visitor (AffiliationVisitor): The visitor

        Raises:
            TypeError: Invalid construction of visitor pattern
        """
        if self == Affiliation.FRIENDLY:
            visitor.visit_friendly()
        elif self == Affiliation.NEUTRAL:
            visitor.visit_neutral()
        elif self == Affiliation.HOSTILE:
            visitor.visit_hostile()
        elif self == Affiliation.UNKNOWN:
            visitor.visit_unknown()
        else:  # pragma: no cover
            msg = "An AffiliationVisitor can only be accepted by an Affiliation"
            raise TypeError(msg)


class AffiliationVisitor(ABC):
    """Affiliation Visitor.

    To be used wherever it is necessary to make a decision based on an affiliation type.
    """

    @abstractmethod
    def visit_friendly(self):  # pragma: no cover
        """Visit a Friendly affiliate."""
        pass

    @abstractmethod
    def visit_neutral(self):  # pragma: no cover
        """Visit a neutral affiliate."""
        pass

    @abstractmethod
    def visit_hostile(self):  # pragma: no cover
        """Visit a hostile affiliate."""
        pass

    @abstractmethod
    def visit_unknown(self):  # pragma: no cover
        """Visit an unknown affiliate."""
        pass
