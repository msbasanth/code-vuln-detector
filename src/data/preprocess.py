"""Preprocessing for C/C++ source code.

Removes comments, preprocessor guards, file headers, and normalizes whitespace
while preserving programmatic structural semantics.
"""

import re


def remove_comments(code: str) -> str:
    """Remove C/C++ comments while respecting string/char literals.

    Handles:
    - Single-line comments: // ...
    - Multi-line comments: /* ... */
    - String literals: "..." (with escaped quotes)
    - Character literals: '...' (with escaped quotes)
    """
    result = []
    i = 0
    n = len(code)

    while i < n:
        # String literal
        if code[i] == '"':
            j = i + 1
            while j < n:
                if code[j] == '\\':
                    j += 2
                elif code[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            result.append(code[i:j])
            i = j

        # Character literal
        elif code[i] == "'":
            j = i + 1
            while j < n:
                if code[j] == '\\':
                    j += 2
                elif code[j] == "'":
                    j += 1
                    break
                else:
                    j += 1
            result.append(code[i:j])
            i = j

        # Single-line comment
        elif i + 1 < n and code[i] == '/' and code[i + 1] == '/':
            # Skip until end of line
            j = i + 2
            while j < n and code[j] != '\n':
                j += 1
            i = j

        # Multi-line comment
        elif i + 1 < n and code[i] == '/' and code[i + 1] == '*':
            j = i + 2
            while j + 1 < n and not (code[j] == '*' and code[j + 1] == '/'):
                j += 1
            i = j + 2

        else:
            result.append(code[i])
            i += 1

    return "".join(result)


# Preprocessor guard patterns to remove
_GUARD_PATTERNS = [
    re.compile(r"^\s*#\s*(?:ifndef|ifdef)\s+(?:OMITBAD|OMITGOOD|INCLUDEMAIN)\s*$", re.MULTILINE),
    re.compile(r"^\s*#\s*endif\s*$", re.MULTILINE),
]


def remove_preprocessor_guards(code: str) -> str:
    """Remove OMITBAD/OMITGOOD/INCLUDEMAIN preprocessor guards."""
    for pattern in _GUARD_PATTERNS:
        code = pattern.sub("", code)
    return code


# Header block pattern — the template-generated file comment
_HEADER_RE = re.compile(
    r"/\*\s*TEMPLATE GENERATED TESTCASE FILE.*?\*/",
    re.DOTALL,
)


def remove_file_header(code: str) -> str:
    """Remove the TEMPLATE GENERATED TESTCASE FILE header block."""
    return _HEADER_RE.sub("", code)


def normalize_whitespace(code: str) -> str:
    """Collapse multiple whitespace characters to single space, strip edges."""
    code = re.sub(r"[ \t]+", " ", code)
    code = re.sub(r"\n\s*\n", "\n", code)
    code = re.sub(r"\n ", "\n", code)
    return code.strip()


def preprocess(code: str) -> str:
    """Full preprocessing pipeline for a code sample."""
    code = remove_file_header(code)
    code = remove_comments(code)
    code = remove_preprocessor_guards(code)
    code = normalize_whitespace(code)
    return code
