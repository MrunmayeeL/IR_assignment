"""Lexer for exact retrieval queries in Lexis."""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class TokenType(Enum):
    TERM = auto()
    PHRASE = auto()
    PROXIMITY_OP = auto()  # /N
    AND = auto()
    OR = auto()
    NOT = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()


@dataclass
class QueryToken:
    type: TokenType
    value: str
    num: int = 0  # For proximity /N


class QueryLexer:
    """Tokenizes Boolean, Phrase, Proximity, and Wildcard query strings."""

    def __init__(self, query_str: str):
        self.text = query_str.strip()
        self.pos = 0
        self.length = len(self.text)

    def tokenize(self) -> List[QueryToken]:
        tokens: List[QueryToken] = []

        while self.pos < self.length:
            char = self.text[self.pos]

            # Whitespace
            if char.isspace():
                self.pos += 1
                continue

            # Parentheses
            if char == "(":
                tokens.append(QueryToken(TokenType.LPAREN, "("))
                self.pos += 1
                continue
            if char == ")":
                tokens.append(QueryToken(TokenType.RPAREN, ")"))
                self.pos += 1
                continue

            # Proximity operator /N (e.g. /5)
            if char == "/":
                self.pos += 1
                num_str = ""
                while self.pos < self.length and self.text[self.pos].isdigit():
                    num_str += self.text[self.pos]
                    self.pos += 1
                if not num_str:
                    num_str = "1"
                tokens.append(QueryToken(TokenType.PROXIMITY_OP, f"/{num_str}", int(num_str)))
                continue

            # Quoted Phrase
            if char == '"':
                self.pos += 1
                phrase_content = ""
                while self.pos < self.length and self.text[self.pos] != '"':
                    phrase_content += self.text[self.pos]
                    self.pos += 1
                if self.pos < self.length and self.text[self.pos] == '"':
                    self.pos += 1  # Consume closing quote
                tokens.append(QueryToken(TokenType.PHRASE, phrase_content))
                continue

            # Word / Term / Operator / Wildcard
            start = self.pos
            while self.pos < self.length and not self.text[self.pos].isspace() and self.text[self.pos] not in "()\"":
                # Check if it hits a proximity operator /N
                if self.text[self.pos] == "/" and self.pos + 1 < self.length and self.text[self.pos + 1].isdigit():
                    break
                self.pos += 1
            word = self.text[start : self.pos]

            upper = word.upper()
            if upper == "AND":
                tokens.append(QueryToken(TokenType.AND, "AND"))
            elif upper == "OR":
                tokens.append(QueryToken(TokenType.OR, "OR"))
            elif upper == "NOT":
                tokens.append(QueryToken(TokenType.NOT, "NOT"))
            else:
                tokens.append(QueryToken(TokenType.TERM, word))

        tokens.append(QueryToken(TokenType.EOF, ""))
        return tokens
