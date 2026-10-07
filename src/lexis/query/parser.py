"""Recursive descent parser for Boolean and exact retrieval queries."""

from dataclasses import dataclass
from typing import List, Optional
from lexis.query.lexer import QueryLexer, QueryToken, TokenType


class Node:
    pass


@dataclass
class TermNode(Node):
    term: str


@dataclass
class WildcardNode(Node):
    pattern: str


@dataclass
class PhraseNode(Node):
    raw_phrase: str


@dataclass
class ProximityNode(Node):
    term1: str
    term2: str
    distance: int


@dataclass
class NotNode(Node):
    child: Node


@dataclass
class AndNode(Node):
    left: Node
    right: Node


@dataclass
class OrNode(Node):
    left: Node
    right: Node


class QueryParser:
    """Parses token stream into an Abstract Syntax Tree (AST)."""

    def __init__(self, tokens: List[QueryToken]):
        self.tokens = tokens
        self.pos = 0

    @classmethod
    def parse_query(cls, query_str: str) -> Node:
        lexer = QueryLexer(query_str)
        tokens = lexer.tokenize()
        parser = cls(tokens)
        return parser.parse()

    def current(self) -> QueryToken:
        return self.tokens[self.pos]

    def consume(self, expected_type: Optional[TokenType] = None) -> QueryToken:
        tok = self.current()
        if expected_type is not None and tok.type != expected_type:
            raise ValueError(f"Expected token {expected_type}, found {tok.type} ({tok.value})")
        if tok.type != TokenType.EOF:
            self.pos += 1
        return tok

    def parse(self) -> Node:
        node = self.parse_or()
        return node

    def parse_or(self) -> Node:
        node = self.parse_and()
        while self.current().type == TokenType.OR:
            self.consume(TokenType.OR)
            right = self.parse_and()
            node = OrNode(left=node, right=right)
        return node

    def parse_and(self) -> Node:
        node = self.parse_not()
        while self.current().type in (TokenType.AND, TokenType.TERM, TokenType.PHRASE, TokenType.LPAREN):
            if self.current().type == TokenType.AND:
                self.consume(TokenType.AND)
            right = self.parse_not()
            node = AndNode(left=node, right=right)
        return node

    def parse_not(self) -> Node:
        if self.current().type == TokenType.NOT:
            self.consume(TokenType.NOT)
            child = self.parse_not()
            return NotNode(child=child)
        return self.parse_atom()

    def parse_atom(self) -> Node:
        tok = self.current()

        if tok.type == TokenType.LPAREN:
            self.consume(TokenType.LPAREN)
            node = self.parse_or()
            if self.current().type == TokenType.RPAREN:
                self.consume(TokenType.RPAREN)
            return node

        if tok.type == TokenType.PHRASE:
            self.consume(TokenType.PHRASE)
            return PhraseNode(raw_phrase=tok.value)

        if tok.type == TokenType.TERM:
            term1 = self.consume(TokenType.TERM).value

            # Check if followed by proximity operator /N
            if self.current().type == TokenType.PROXIMITY_OP:
                op_tok = self.consume(TokenType.PROXIMITY_OP)
                if self.current().type != TokenType.TERM:
                    raise ValueError(f"Proximity operator {op_tok.value} must be followed by a term")
                term2 = self.consume(TokenType.TERM).value
                return ProximityNode(term1=term1, term2=term2, distance=op_tok.num)

            # Check if wildcard
            if "*" in term1:
                return WildcardNode(pattern=term1)

            return TermNode(term=term1)

        raise ValueError(f"Unexpected token in query: {tok}")
