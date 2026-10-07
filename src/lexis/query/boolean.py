"""Boolean query evaluator over AST nodes."""

from typing import Set, List
from lexis.index.positional_index import PositionalIndex
from lexis.preprocessing.tokenizer import Tokenizer
from lexis.query.parser import (
    Node,
    TermNode,
    PhraseNode,
    ProximityNode,
    WildcardNode,
    NotNode,
    AndNode,
    OrNode,
)
from lexis.query.phrase import PhraseEvaluator
from lexis.query.proximity import ProximityEvaluator
from lexis.query.wildcard import WildcardEvaluator


class BooleanEvaluator:
    """Evaluates Boolean AST trees into matching document IDs."""

    def __init__(self, index: PositionalIndex, tokenizer: Tokenizer):
        self.index = index
        self.tokenizer = tokenizer
        self.phrase_evaluator = PhraseEvaluator(index, tokenizer)
        self.proximity_evaluator = ProximityEvaluator(index, tokenizer)
        self.wildcard_evaluator = WildcardEvaluator(index)

    def get_all_doc_ids(self) -> Set[int]:
        """Set of all active (non-deleted) integer doc IDs in index."""
        deleted = self.index.document_registry.deleted_doc_ids
        return {
            doc_id
            for doc_id in self.index.document_registry.doc_id_to_name.keys()
            if doc_id not in deleted
        }

    def evaluate_ast(self, node: Node) -> Set[int]:
        if isinstance(node, TermNode):
            terms = self.tokenizer.tokenize_terms(node.term)
            if not terms:
                return set()
            term = terms[0]
            plist = self.index.get_postings(term)
            return set(plist.doc_ids()) if plist else set()

        elif isinstance(node, PhraseNode):
            return set(self.phrase_evaluator.evaluate(node.raw_phrase))

        elif isinstance(node, ProximityNode):
            return set(
                self.proximity_evaluator.evaluate(
                    term1_raw=node.term1,
                    distance=node.distance,
                    term2_raw=node.term2,
                )
            )

        elif isinstance(node, WildcardNode):
            return set(self.wildcard_evaluator.evaluate(node.pattern))

        elif isinstance(node, AndNode):
            left_docs = self.evaluate_ast(node.left)
            # Short-circuit if left is empty
            if not left_docs:
                return set()
            right_docs = self.evaluate_ast(node.right)
            return left_docs.intersection(right_docs)

        elif isinstance(node, OrNode):
            left_docs = self.evaluate_ast(node.left)
            right_docs = self.evaluate_ast(node.right)
            return left_docs.union(right_docs)

        elif isinstance(node, NotNode):
            child_docs = self.evaluate_ast(node.child)
            all_docs = self.get_all_doc_ids()
            return all_docs.difference(child_docs)

        else:
            raise ValueError(f"Unknown AST node type: {type(node)}")

    def execute(self, ast: Node) -> List[str]:
        """Evaluate AST and return sorted document names (e.g. ['doc_00001', 'doc_00005'])."""
        doc_ids = self.evaluate_ast(ast)
        doc_names = [
            self.index.get_doc_name(did)
            for did in doc_ids
            if self.index.get_doc_name(did) is not None
        ]
        return sorted(doc_names)
