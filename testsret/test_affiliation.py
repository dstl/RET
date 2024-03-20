"""Tests for affiliation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import TestCase

from ret.agents.affiliation import Affiliation, AffiliationVisitor

if TYPE_CHECKING:
    from typing import Optional


class TestAffiliationVisitor(TestCase):
    """Test cases for the base AffiliationVisitor."""

    class TestAffiliationVisitor(AffiliationVisitor):
        """Simple implementation of the AffiliationVisitor, to record visit."""

        visited: Optional[Affiliation]

        def visit_friendly(self):
            """Visit a friendly affiliation."""
            self.visited = Affiliation.FRIENDLY

        def visit_neutral(self):
            """Visit a neutral affiliation."""
            self.visited = Affiliation.NEUTRAL

        def visit_hostile(self):
            """Visit a hostile affiliation."""
            self.visited = Affiliation.HOSTILE

        def visit_unknown(self):
            """Visit an unknown affiliation."""
            self.visited = Affiliation.UNKNOWN

    def test_visit_friendly(self):
        """Test functionality of accepting visitor on Friendly."""
        visitor = self.TestAffiliationVisitor()
        Affiliation.FRIENDLY.accept(visitor)
        assert visitor.visited == Affiliation.FRIENDLY

    def test_visit_neutral(self):
        """Test functionality of accepting visitor on Neutral."""
        visitor = self.TestAffiliationVisitor()
        Affiliation.NEUTRAL.accept(visitor)
        assert visitor.visited == Affiliation.NEUTRAL

    def test_visit_hostile(self):
        """Test functionality of accepting visitor on Hostile."""
        visitor = self.TestAffiliationVisitor()
        Affiliation.HOSTILE.accept(visitor)
        assert visitor.visited == Affiliation.HOSTILE

    def test_visit_unknown(self):
        """Test functionality of accepting visitor on Unknown."""
        visitor = self.TestAffiliationVisitor()
        Affiliation.UNKNOWN.accept(visitor)
        assert visitor.visited == Affiliation.UNKNOWN
