"""Porter Stemmer implementation for Lexis (Martin Porter, 1980).

Completely self-contained, deterministic, and configurable.
"""


class PorterStemmer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def stem(self, word: str) -> str:
        """Stem a word using the Porter stemmer algorithm if enabled."""
        if not self.enabled or len(word) <= 2:
            return word

        w = word.lower()
        if not w.isalpha():
            return w

        # Step 1a
        w = self._step_1a(w)
        # Step 1b
        w = self._step_1b(w)
        # Step 1c
        w = self._step_1c(w)
        # Step 2
        w = self._step_2(w)
        # Step 3
        w = self._step_3(w)
        # Step 4
        w = self._step_4(w)
        # Step 5a
        w = self._step_5a(w)
        # Step 5b
        w = self._step_5b(w)
        return w

    def _is_consonant(self, word: str, i: int) -> bool:
        letter = word[i]
        if letter in "aeiou":
            return False
        if letter == "y":
            if i == 0:
                return True
            return not self._is_consonant(word, i - 1)
        return True

    def _measure(self, stem: str) -> int:
        """Measure m in [C](VC)^m[V]."""
        if not stem:
            return 0
        c_v_seq = []
        for i in range(len(stem)):
            c = self._is_consonant(stem, i)
            if not c_v_seq or c_v_seq[-1] != c:
                c_v_seq.append(c)
        # c_v_seq is alternating sequence of True (C) and False (V)
        # Count VC transitions
        m = 0
        for i in range(len(c_v_seq) - 1):
            if not c_v_seq[i] and c_v_seq[i + 1]:  # V followed by C
                m += 1
        return m

    def _contains_vowel(self, stem: str) -> bool:
        for i in range(len(stem)):
            if not self._is_consonant(stem, i):
                return True
        return False

    def _ends_double_consonant(self, stem: str) -> bool:
        if len(stem) < 2:
            return False
        return stem[-1] == stem[-2] and self._is_consonant(stem, len(stem) - 1)

    def _cvc(self, stem: str) -> bool:
        """Check if stem ends with consonant, vowel, consonant (not w, x, y)."""
        if len(stem) < 3:
            return False
        if (
            self._is_consonant(stem, len(stem) - 1)
            and not self._is_consonant(stem, len(stem) - 2)
            and self._is_consonant(stem, len(stem) - 3)
        ):
            if stem[-1] not in "wxy":
                return True
        return False

    def _step_1a(self, word: str) -> str:
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-2]
        if word.endswith("ss"):
            return word
        if word.endswith("s"):
            return word[:-1]
        return word

    def _step_1b(self, word: str) -> str:
        if word.endswith("eed"):
            stem = word[:-3]
            if self._measure(stem) > 0:
                return word[:-1]
            return word

        matched = False
        stem = ""
        if word.endswith("ed"):
            stem = word[:-2]
            if self._contains_vowel(stem):
                word = stem
                matched = True
        elif word.endswith("ing"):
            stem = word[:-3]
            if self._contains_vowel(stem):
                word = stem
                matched = True

        if matched:
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                return word + "e"
            if self._ends_double_consonant(word) and word[-1] not in "lsz":
                return word[:-1]
            if self._measure(word) == 1 and self._cvc(word):
                return word + "e"
        return word

    def _step_1c(self, word: str) -> str:
        if word.endswith("y"):
            stem = word[:-1]
            if self._contains_vowel(stem):
                return stem + "i"
        return word

    def _step_2(self, word: str) -> str:
        pairs = [
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("abli", "able"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
        ]
        for suffix, repl in pairs:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self._measure(stem) > 0:
                    return stem + repl
                return word
        return word

    def _step_3(self, word: str) -> str:
        pairs = [
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ]
        for suffix, repl in pairs:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self._measure(stem) > 0:
                    return stem + repl
                return word
        return word

    def _step_4(self, word: str) -> str:
        suffixes = [
            "al",
            "ance",
            "ence",
            "er",
            "ic",
            "able",
            "ible",
            "ant",
            "ement",
            "ment",
            "ent",
            "ou",
            "ism",
            "ate",
            "iti",
            "ous",
            "ive",
            "ize",
        ]
        for suffix in suffixes:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self._measure(stem) > 1:
                    return stem
                return word
        if word.endswith("sion") or word.endswith("tion"):
            stem = word[:-3]  # leaves s or t
            if self._measure(stem) > 1:
                return stem
        return word

    def _step_5a(self, word: str) -> str:
        if word.endswith("e"):
            stem = word[:-1]
            m = self._measure(stem)
            if m > 1:
                return stem
            if m == 1 and not self._cvc(stem):
                return stem
        return word

    def _step_5b(self, word: str) -> str:
        if self._measure(word) > 1 and self._ends_double_consonant(word) and word.endswith("l"):
            return word[:-1]
        return word
