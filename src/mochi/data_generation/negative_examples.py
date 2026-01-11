"""Negative example generation for training data.

Generates training examples that teach the model to avoid common hallucinations
by showing incorrect API calls and their corrections.

Law compliance:
- L-hallucination-prevent: Train model to recognize and avoid common mistakes
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from mochi.data_generation.alpaca_converter import AlpacaExample


# Common API hallucination patterns: (correct method, incorrect variations)
# Based on observed model hallucinations and evaluation failures
COMMON_HALLUCINATIONS: dict[str, list[str]] = {
    # DuckDB patterns - HIGH PRIORITY (from golden dataset failures)
    "all": ["getAll", "findAll", "selectAll", "queryAll", "fetchAll", "query", "execute"],
    "run": ["execute", "exec", "runQuery", "executeQuery", "query"],
    "prepare": ["createStatement", "prepareStatement", "stmt", "statement"],
    # Query patterns
    "query": ["runQuery", "executeQuery", "rawQuery"],
    "select": ["get", "find", "fetch"],
    # String/text patterns - HIGH PRIORITY (from golden dataset failures)
    "split": ["tokenize", "splitText", "splitString"],
    "trim": ["strip", "trimText"],
    "toLowerCase": ["lower", "toLower", "downcase", "lowerCase", "lowercase"],
    "toUpperCase": ["upper", "toUpper", "upcase", "upperCase", "uppercase"],
    # Array patterns - HIGH PRIORITY (from golden dataset failures)
    "filter": ["where", "filterBy", "select"],
    "map": ["transform", "mapTo", "convert"],
    "find": ["get", "findOne", "first", "locate"],
    "forEach": ["each", "iterate", "loop"],
    "includes": ["contains", "has", "hasItem", "isIn", "contain"],
    "indexOf": ["findIndex", "index", "getIndex"],
    # Promise patterns - HIGH PRIORITY (from golden dataset failures)
    "then": ["onSuccess", "success", "done", "onResolve", "resolve"],
    "catch": ["onError", "error", "fail", "onFail", "onReject"],
    "finally": ["always", "complete", "finished"],
    # File system patterns - HIGH PRIORITY (from golden dataset failures)
    "readFile": ["read", "loadFile", "getFile", "load", "open"],
    "writeFile": ["write", "saveFile", "putFile", "save"],
    "exists": ["fileExists", "isFile", "hasFile"],
    "readdir": ["listDir", "readDir", "ls", "listFiles"],
    # Path patterns
    "join": ["combine", "concat", "merge"],
    "resolve": ["absolute", "toAbsolute", "fullPath"],
    # JSON patterns
    "parse": ["fromJson", "parseJson", "decode"],
    "stringify": ["toJson", "encode", "serialize"],
}

# Domain-specific hallucinations for kiri codebase - HIGH PRIORITY
KIRI_SPECIFIC_HALLUCINATIONS: dict[str, list[str]] = {
    # kiri uses specific patterns (from golden dataset failures)
    "isStopWord": ["checkStopWord", "isStop", "stopWord", "isStopword", "checkStop"],
    "scanDirectory": ["scan", "walkDir", "readDir", "listFiles", "scanDir"],
    "extractKeywords": ["getKeywords", "keywords", "findKeywords"],
    "buildIndex": ["createIndex", "index", "makeIndex"],
    "processFile": ["process", "handleFile", "parseFile"],
    "analyze": ["parse", "process", "check"],
    # Database patterns (DuckDBClient)
    "all": ["query", "select", "getRows", "fetch", "findAll", "getAll"],
    "run": ["execute", "exec", "query", "runQuery"],
    "prepare": ["statement", "createStatement", "prepareStatement"],
    # P1: 追加パターン (kiri固有)
    # Token/Text processing
    "tokenize": ["split", "parse", "tokens", "getTokens"],
    "normalize": ["clean", "sanitize", "format", "process"],
    "lemmatize": ["stem", "root", "base"],
    # File handling
    "readFileContent": ["readFile", "getContent", "load", "read"],
    "writeResult": ["save", "write", "output", "store"],
    # Index operations
    "addDocument": ["add", "insert", "index", "put"],
    "searchIndex": ["search", "query", "find", "lookup"],
    "updateIndex": ["update", "refresh", "rebuild"],
    # Scanner patterns
    "walkDirectory": ["walk", "traverse", "scan", "iterate"],
    "filterFiles": ["filter", "select", "match", "include"],
}

# Exact method name pairs for strict training
# Format: (exact_correct, common_synonym_to_reject)
EXACT_METHOD_PAIRS: list[tuple[str, str, str]] = [
    # (correct, wrong, context_type)
    ("toLowerCase", "lower", "String"),
    ("toLowerCase", "toLower", "String"),
    ("toLowerCase", "lowerCase", "String"),
    ("toUpperCase", "upper", "String"),
    ("toUpperCase", "toUpper", "String"),
    ("includes", "contains", "Array"),
    ("includes", "has", "Array"),
    ("readFile", "read", "FileSystem"),
    ("writeFile", "write", "FileSystem"),
    ("then", "success", "Promise"),
    ("catch", "error", "Promise"),
    ("isStopWord", "isStop", "TokenAnalyzer"),
    ("isStopWord", "stopWord", "TokenAnalyzer"),
    ("scanDirectory", "scan", "Scanner"),
    ("prepare", "statement", "Database"),
    ("all", "query", "DuckDBClient"),
    ("all", "execute", "DuckDBClient"),
    ("run", "execute", "DuckDBClient"),
    # P1: 追加ペア (kiri固有)
    ("all", "getAll", "DuckDBClient"),
    ("all", "findAll", "DuckDBClient"),
    ("run", "exec", "DuckDBClient"),
    ("tokenize", "split", "Tokenizer"),
    ("normalize", "clean", "TextProcessor"),
    ("addDocument", "add", "SearchIndex"),
    ("searchIndex", "search", "SearchIndex"),
    ("walkDirectory", "walk", "FileScanner"),
    ("filterFiles", "filter", "FileScanner"),
    ("extractKeywords", "getKeywords", "KeywordExtractor"),
    ("buildIndex", "createIndex", "IndexBuilder"),
]


@dataclass
class NegativeExample:
    """A training example showing incorrect code and correction."""

    # Context showing the bad code
    bad_code: str
    # Corrected code
    good_code: str
    # What was wrong
    error_description: str
    # The method that was hallucinated
    hallucinated_method: str
    # The correct method
    correct_method: str


class NegativeExampleGenerator:
    """Generate negative training examples for hallucination prevention.

    Creates examples that show common API mistakes and their corrections,
    teaching the model to avoid these patterns.

    Usage:
        generator = NegativeExampleGenerator()
        examples = generator.generate_from_code(code_with_method_calls)

    Or:
        examples = generator.generate_synthetic(num_examples=50)
    """

    def __init__(
        self,
        include_kiri_specific: bool = True,
        hallucination_patterns: dict[str, list[str]] | None = None,
    ) -> None:
        """Initialize generator.

        Args:
            include_kiri_specific: Include kiri-specific hallucination patterns
            hallucination_patterns: Custom hallucination patterns to use
        """
        self.patterns = dict(COMMON_HALLUCINATIONS)
        if include_kiri_specific:
            self.patterns.update(KIRI_SPECIFIC_HALLUCINATIONS)
        if hallucination_patterns:
            self.patterns.update(hallucination_patterns)

    def generate_correction_examples(
        self,
        num_examples: int = 50,
    ) -> list[AlpacaExample]:
        """Generate synthetic correction examples.

        Creates examples that show incorrect API usage and ask the model
        to identify and fix the mistake.

        Args:
            num_examples: Number of examples to generate

        Returns:
            List of AlpacaExample for training
        """
        examples = []
        patterns = list(self.patterns.items())

        for _ in range(num_examples):
            correct, incorrects = random.choice(patterns)
            incorrect = random.choice(incorrects)

            # Create various code patterns
            pattern_type = random.choice([
                "simple_call",
                "chained_call",
                "async_call",
                "callback",
            ])

            bad_code, good_code = self._generate_code_pair(
                incorrect, correct, pattern_type
            )

            # Create correction instruction
            instruction = random.choice([
                "Fix the incorrect method call in this code:",
                "Correct the API usage in this code:",
                "The following code has an incorrect method name. Fix it:",
                "This code uses a non-existent method. Correct it:",
            ])

            examples.append(AlpacaExample(
                instruction=instruction,
                input=bad_code,
                output=good_code,
            ))

        return examples

    def generate_identification_examples(
        self,
        num_examples: int = 30,
    ) -> list[AlpacaExample]:
        """Generate examples for identifying incorrect API usage.

        Creates examples that ask the model to identify what's wrong
        with the code, teaching it to recognize hallucinated patterns.

        Args:
            num_examples: Number of examples to generate

        Returns:
            List of AlpacaExample for training
        """
        examples = []
        patterns = list(self.patterns.items())

        for _ in range(num_examples):
            correct, incorrects = random.choice(patterns)
            incorrect = random.choice(incorrects)

            pattern_type = random.choice([
                "simple_call",
                "chained_call",
                "async_call",
            ])

            bad_code, _ = self._generate_code_pair(incorrect, correct, pattern_type)

            instruction = random.choice([
                "What is wrong with this code?",
                "Identify the API error in this code:",
                "This code has an incorrect method call. What is it?",
                "Find the non-existent method in this code:",
            ])

            # Output explains the error
            output = f"The method `{incorrect}` does not exist. The correct method is `{correct}`."

            examples.append(AlpacaExample(
                instruction=instruction,
                input=bad_code,
                output=output,
            ))

        return examples

    def generate_context_constrained_examples(
        self,
        num_examples: int = 50,
    ) -> list[AlpacaExample]:
        """Generate examples with explicit context constraints.

        Creates examples that include a context block with available methods,
        and the output must use only those methods.

        Args:
            num_examples: Number of examples to generate

        Returns:
            List of AlpacaExample for training
        """
        examples = []
        patterns = list(self.patterns.items())

        for _ in range(num_examples):
            correct, incorrects = random.choice(patterns)

            # Create a context block with available methods
            available_methods = [correct]
            # Add some other random correct methods
            for other_correct, _ in random.sample(patterns, min(3, len(patterns))):
                if other_correct != correct:
                    available_methods.append(other_correct)

            context = "// Available methods:\n"
            for method in available_methods:
                context += f"//   {method}()\n"

            # Create code that uses the correct method
            pattern_type = random.choice(["simple_call", "chained_call", "async_call"])
            _, good_code = self._generate_code_pair(
                random.choice(incorrects), correct, pattern_type
            )

            instruction = random.choice([
                "Complete the code using ONLY the methods listed in the context:",
                "Write code using the available methods:",
                "Using only the provided API, complete this code:",
            ])

            # Input includes context and partial code
            partial_code = good_code.rsplit(".", 1)[0] + "."

            full_input = f"{context}\n{partial_code}"

            # Output is the completed part
            completion = good_code.split(".")[-1]

            examples.append(AlpacaExample(
                instruction=instruction,
                input=full_input,
                output=completion,
            ))

        return examples

    def _generate_code_pair(
        self,
        incorrect_method: str,
        correct_method: str,
        pattern_type: str,
    ) -> tuple[str, str]:
        """Generate a pair of incorrect/correct code snippets.

        Args:
            incorrect_method: The hallucinated method name
            correct_method: The correct method name
            pattern_type: Type of code pattern to generate

        Returns:
            Tuple of (bad_code, good_code)
        """
        # Variable names to use
        var_names = ["db", "client", "result", "data", "items", "str", "arr", "file"]
        var = random.choice(var_names)

        # Arguments
        args_options = [
            '("query")',
            "(data)",
            "(items)",
            '("value")',
            "(callback)",
            "(options)",
            "()",
        ]
        args = random.choice(args_options)

        if pattern_type == "simple_call":
            bad = f"{var}.{incorrect_method}{args};"
            good = f"{var}.{correct_method}{args};"

        elif pattern_type == "chained_call":
            chain_method = random.choice(["then", "catch", "map", "filter"])
            bad = f"{var}.{incorrect_method}{args}.{chain_method}(x => x);"
            good = f"{var}.{correct_method}{args}.{chain_method}(x => x);"

        elif pattern_type == "async_call":
            bad = f"const result = await {var}.{incorrect_method}{args};"
            good = f"const result = await {var}.{correct_method}{args};"

        elif pattern_type == "callback":
            bad = f'{var}.{incorrect_method}{args.replace(")", ", (err, res) => {}")};'
            good = f'{var}.{correct_method}{args.replace(")", ", (err, res) => {}")};'

        else:
            bad = f"{var}.{incorrect_method}{args};"
            good = f"{var}.{correct_method}{args};"

        return bad, good

    def generate_exact_match_examples(
        self,
        num_examples: int = 100,
    ) -> list[AlpacaExample]:
        """Generate examples that enforce exact method name usage.

        Creates examples that explicitly show:
        1. The context has method X
        2. The user might think of synonym Y
        3. But the output MUST use X, not Y

        Args:
            num_examples: Number of examples to generate

        Returns:
            List of AlpacaExample for training
        """
        examples = []

        for _ in range(num_examples):
            correct, wrong, context_type = random.choice(EXACT_METHOD_PAIRS)

            # Create context with the correct method
            context = f"// Methods on {context_type}:\n//   {correct}(): result"

            # Create instruction emphasizing exact match
            instruction = random.choice([
                f"Complete the code. Use ONLY methods from the context. Do NOT use '{wrong}':",
                f"Complete using the exact method name from context (not '{wrong}'):",
                f"The context shows '{correct}'. Complete the code (do not use '{wrong}'):",
                f"Available: {correct}(). NOT available: {wrong}(). Complete:",
            ])

            # Input is partial code
            var_name = random.choice(["obj", "client", "instance", "x", "data"])
            input_text = f"{context}\n\nconst result = {var_name}."

            # Output must use the correct method
            output = f"{correct}();"

            examples.append(AlpacaExample(
                instruction=instruction,
                input=input_text,
                output=output,
            ))

        return examples

    def generate_rejection_examples(
        self,
        num_examples: int = 50,
    ) -> list[AlpacaExample]:
        """Generate examples where model must reject using unavailable methods.

        Teaches the model to recognize when a method is NOT in context
        and to use the correct alternative.

        Args:
            num_examples: Number of examples to generate

        Returns:
            List of AlpacaExample for training
        """
        examples = []

        for _ in range(num_examples):
            correct, wrong, context_type = random.choice(EXACT_METHOD_PAIRS)

            # Context only has the correct method
            context = f"// Methods on {context_type}:\n//   {correct}(arg): result\n// Note: {wrong}() does NOT exist"

            instruction = "Complete the code using only available methods:"

            var_name = random.choice(["obj", "client", "instance"])
            input_text = f"{context}\n\n// I want to use something like {wrong}...\nconst result = {var_name}."

            # Output explains the correct choice
            output = f"{correct}(arg);  // Use {correct}, not {wrong}"

            examples.append(AlpacaExample(
                instruction=instruction,
                input=input_text,
                output=output,
            ))

        return examples

    def generate_all(
        self,
        correction_count: int = 50,
        identification_count: int = 30,
        constrained_count: int = 50,
        exact_match_count: int = 100,
        rejection_count: int = 50,
    ) -> list[AlpacaExample]:
        """Generate all types of negative examples.

        Args:
            correction_count: Number of correction examples
            identification_count: Number of identification examples
            constrained_count: Number of context-constrained examples
            exact_match_count: Number of exact match examples
            rejection_count: Number of rejection examples

        Returns:
            Combined list of all example types
        """
        all_examples = []

        all_examples.extend(self.generate_correction_examples(correction_count))
        all_examples.extend(self.generate_identification_examples(identification_count))
        all_examples.extend(self.generate_context_constrained_examples(constrained_count))
        all_examples.extend(self.generate_exact_match_examples(exact_match_count))
        all_examples.extend(self.generate_rejection_examples(rejection_count))

        # Shuffle to mix example types
        random.shuffle(all_examples)

        return all_examples


def generate_negative_training_data(
    output_path: str,
    num_examples: int = 100,
    include_kiri: bool = True,
) -> int:
    """Convenience function to generate negative training data.

    Args:
        output_path: Path to save the JSONL output
        num_examples: Total number of examples to generate
        include_kiri: Include kiri-specific patterns

    Returns:
        Number of examples generated
    """
    from pathlib import Path
    from mochi.data_generation.alpaca_converter import AlpacaConverter

    generator = NegativeExampleGenerator(include_kiri_specific=include_kiri)

    # Distribute examples across types
    correction = num_examples // 3
    identification = num_examples // 4
    constrained = num_examples - correction - identification

    examples = generator.generate_all(
        correction_count=correction,
        identification_count=identification,
        constrained_count=constrained,
    )

    # Save using the same format as other training data
    converter = AlpacaConverter()
    converter.to_jsonl(examples, output_path)

    return len(examples)
