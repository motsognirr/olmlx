"""Curated micro-sets for quality benchmarking.

Three small prompt sets for measuring absolute task quality alongside
regression-vs-golden drift:

- ``GSM8K_MINI`` — 20 grade-school math word problems. Graded with
  ``numeric``; the final integer is pulled out of the model's output
  (``#### N``, ``\\boxed{N}``, or trailing number). Source: GSM8K test
  split (MIT License; Cobbe et al., 2021, https://arxiv.org/abs/2110.14168).

- ``MMLU_MINI`` — 20 four-choice questions spanning four subjects.
  Graded with ``regex_match`` on a trailing "Answer: X" pattern. Source:
  MMLU (MIT License; Hendrycks et al., 2020,
  https://arxiv.org/abs/2009.03300).

- ``HUMANEVAL_MINI`` — 10 coding problems with unit-test check functions.
  Graded with ``code_exec`` in a sandboxed subprocess (opt-in behind
  ``--enable-code-exec``). Source: OpenAI HumanEval (MIT License; Chen
  et al., 2021, https://arxiv.org/abs/2107.03374).

Problems are bundled as literal Python to avoid runtime downloads and
keep test iteration fast. All sets are intentionally small so a run
finishes in minutes on a laptop.
"""

from __future__ import annotations

from olmlx.bench.prompts import BenchPrompt


def _numeric(
    name: str, question: str, answer: int, max_tokens: int = 512
) -> BenchPrompt:
    return BenchPrompt(
        name=name,
        category="gsm8k",
        messages=[
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Think step by step, then write the final answer on its own "
                    "line as '#### <number>'."
                ),
            }
        ],
        max_tokens=max_tokens,
        grader="numeric",
        expected={"answer": answer, "tol": 0.0},
    )


GSM8K_MINI: list[BenchPrompt] = [
    _numeric(
        "gsm8k-01-janet-eggs",
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
        "morning and bakes muffins for her friends every day with four. She sells "
        "the remainder at the farmers' market daily for $2 per fresh duck egg. "
        "How much in dollars does she make every day at the farmers' market?",
        18,
    ),
    _numeric(
        "gsm8k-02-robe-bolts",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How "
        "many bolts in total does it take?",
        3,
    ),
    _numeric(
        "gsm8k-03-weng-babysitting",
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
        "minutes of babysitting. How much did she earn?",
        10,
    ),
    _numeric(
        "gsm8k-04-betty-wallet",
        "Betty is saving money for a new wallet which costs $100. Betty has only "
        "half of the money she needs. Her parents decided to give her $15 for "
        "that purpose, and her grandparents twice as much as her parents. How "
        "much more money does Betty need to buy the wallet?",
        5,
    ),
    _numeric(
        "gsm8k-05-julie-book",
        "Julie is reading a 120-page book. Yesterday, she was able to read 12 "
        "pages and today, she read twice as many pages as yesterday. If she "
        "wants to read half of the remaining pages tomorrow, how many pages "
        "should she read?",
        42,
    ),
    _numeric(
        "gsm8k-06-james-sprints",
        "James decides to run 3 sprints 3 times a week. He runs 60 meters each "
        "sprint. How many total meters does he run a week?",
        540,
    ),
    _numeric(
        "gsm8k-07-ken-gift-box",
        "Ken created a care package to send to his brother, who was away at "
        "boarding school. Ken placed a box on a scale, and then he poured into "
        "the box enough jelly beans to bring the weight to 2 pounds. Then, he "
        "added enough brownies to cause the weight to triple. Next, he added "
        "another 2 pounds of jelly beans. And finally, he added enough gummy "
        "worms to double the weight once again. What was the final weight of "
        "the box of goodies, in pounds?",
        16,
    ),
    _numeric(
        "gsm8k-08-alexis-shopping",
        "Alexis is applying for a new job and bought a new set of business clothes "
        "to wear to the interview. She went to a department store with a budget "
        "of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a "
        "suit coat, $11 on socks, and $18 on a belt. She also purchased a pair "
        "of shoes, but lost the receipt for them. She has $16 left from her "
        "budget. How much did Alexis pay for the shoes?",
        41,
    ),
    _numeric(
        "gsm8k-09-tim-quiz",
        "Tim has 30 toads. Jim has 20 more toads than Tim does. Sarah has twice "
        "as many toads as Jim does. How many toads does Sarah have?",
        100,
    ),
    _numeric(
        "gsm8k-10-marcy-lipstick",
        # Reworded from the original GSM8K-style phrasing — "1/3 of that time
        # combing it" was genuinely ambiguous between additive (12 + 4 = 16)
        # and partitive (combing is *part of* the 12). "Then she spends an
        # additional ..." forces the additive reading.
        "Marcy spends 12 minutes petting her cat. Then she spends an additional "
        "1/3 of that time combing it. How many minutes does she spend with her "
        "cat total?",
        16,
    ),
    _numeric(
        "gsm8k-11-farm-chickens",
        "A farmer has 46 chickens. Each chicken produces 6 eggs a week. If he "
        "sells a dozen eggs for $3, how much money would he make in 8 weeks?",
        552,
    ),
    _numeric(
        "gsm8k-12-book-pages",
        "A book has 3 chapters. The first chapter is 66 pages long, the second "
        "chapter is 35 pages long, and the third chapter is 24 pages long. How "
        "many more pages does the first chapter have than the second and third "
        "chapters combined?",
        7,
    ),
    _numeric(
        "gsm8k-13-john-cars",
        "John cleans 3 cars. The first car takes 40 minutes to clean. The next "
        "two take twice as long each. How long did he spend cleaning total, in "
        "minutes?",
        200,
    ),
    _numeric(
        "gsm8k-14-pizza-slices",
        "A pizza has 8 slices. Tom eats 3 slices and Jerry eats 3 slices. What "
        "percent of the pizza is left? Answer as an integer (e.g. '50' for 50%).",
        25,
    ),
    _numeric(
        "gsm8k-15-bus-riders",
        "A bus has 50 seats. 3/5 of the seats are occupied. How many seats are empty?",
        20,
    ),
    _numeric(
        "gsm8k-16-savings",
        "Maya saves $20 per week. After how many weeks will she have saved $260?",
        13,
    ),
    _numeric(
        "gsm8k-17-pencils",
        "A box of pencils costs $3.60 and holds 12 pencils. How much does a "
        "single pencil cost, in cents?",
        30,
    ),
    _numeric(
        "gsm8k-18-train-speed",
        "A train travels 60 miles in 1.5 hours at a constant speed. If it keeps "
        "the same speed, how many miles will it travel in 4 hours?",
        160,
    ),
    _numeric(
        "gsm8k-19-apples",
        "Sam has 3 times as many apples as Lily. Together they have 48 apples. "
        "How many apples does Lily have?",
        12,
    ),
    _numeric(
        "gsm8k-20-rectangle",
        "A rectangle has a length of 12 and a width of 5. What is its area?",
        60,
    ),
]


def _mmlu(
    name: str,
    subject: str,
    question: str,
    choices: tuple[str, str, str, str],
    answer: str,
) -> BenchPrompt:
    a, b, c, d = choices
    content = (
        f"Question: {question}\n"
        f"A) {a}\nB) {b}\nC) {c}\nD) {d}\n\n"
        "Respond with a single line at the end: 'Answer: <letter>'."
    )
    return BenchPrompt(
        name=name,
        category=f"mmlu-{subject}",
        messages=[{"role": "user", "content": content}],
        max_tokens=128,
        grader="regex_match",
        expected={
            "pattern": r"(?i)answer[:\s]*([A-D])",
            "group": 1,
            "answer": answer,
        },
    )


MMLU_MINI: list[BenchPrompt] = [
    # elementary_mathematics
    _mmlu(
        "mmlu-01-em-multiplication",
        "elem-math",
        "What is 24 * 7?",
        ("140", "158", "168", "174"),
        "C",
    ),
    _mmlu(
        "mmlu-02-em-fraction",
        "elem-math",
        "Which fraction is largest?",
        ("1/2", "2/3", "3/5", "4/9"),
        "B",
    ),
    _mmlu(
        "mmlu-03-em-perimeter",
        "elem-math",
        "A square has a side length of 5 cm. What is its perimeter?",
        ("10 cm", "20 cm", "25 cm", "30 cm"),
        "B",
    ),
    _mmlu(
        "mmlu-04-em-division",
        "elem-math",
        "What is 144 divided by 12?",
        ("10", "11", "14", "12"),
        "D",
    ),
    _mmlu(
        "mmlu-05-em-percent",
        "elem-math",
        "What is 25% of 80?",
        ("16", "18", "20", "25"),
        "C",
    ),
    # world_history
    _mmlu(
        "mmlu-06-hist-ww2",
        "world-history",
        "In what year did World War II end?",
        ("1943", "1947", "1950", "1945"),
        "D",
    ),
    _mmlu(
        "mmlu-07-hist-fr-rev",
        "world-history",
        "The French Revolution began in which year?",
        ("1776", "1799", "1812", "1789"),
        "D",
    ),
    _mmlu(
        "mmlu-08-hist-berlin",
        "world-history",
        "The Berlin Wall fell in which year?",
        ("1985", "1987", "1989", "1991"),
        "C",
    ),
    _mmlu(
        "mmlu-09-hist-rome",
        "world-history",
        "The Western Roman Empire fell in which year?",
        ("410 AD", "476 AD", "500 AD", "527 AD"),
        "B",
    ),
    _mmlu(
        "mmlu-10-hist-printing-press",
        "world-history",
        "Who is credited with inventing the movable-type printing press in Europe?",
        ("Johannes Kepler", "Johannes Gutenberg", "Martin Luther", "Leonardo da Vinci"),
        "B",
    ),
    # science
    _mmlu(
        "mmlu-11-sci-photosynthesis",
        "science",
        "Which gas do plants primarily absorb from the atmosphere during photosynthesis?",
        ("Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"),
        "C",
    ),
    _mmlu(
        "mmlu-12-sci-water",
        "science",
        "What is the chemical formula for water?",
        ("H2O", "HO2", "H3O", "H2O2"),
        "A",
    ),
    _mmlu(
        "mmlu-13-sci-light",
        "science",
        "What is the approximate speed of light in vacuum?",
        (
            "3 × 10^5 m/s",
            "3 × 10^7 m/s",
            "3 × 10^8 m/s",
            "3 × 10^10 m/s",
        ),
        "C",
    ),
    _mmlu(
        "mmlu-14-sci-dna",
        "science",
        "Which molecule carries genetic information in living cells?",
        ("DNA", "ATP", "Glucose", "Lipid"),
        "A",
    ),
    _mmlu(
        "mmlu-15-sci-planet",
        "science",
        "Which is the largest planet in our solar system?",
        ("Earth", "Mars", "Saturn", "Jupiter"),
        "D",
    ),
    # programming
    _mmlu(
        "mmlu-16-prog-big-o",
        "programming",
        "What is the worst-case time complexity of binary search on a sorted "
        "array of n elements?",
        ("O(1)", "O(log n)", "O(n)", "O(n log n)"),
        "B",
    ),
    _mmlu(
        "mmlu-17-prog-python-list",
        "programming",
        "In Python, what is the result of list(range(5))?",
        (
            "[0, 1, 2, 3, 4]",
            "[1, 2, 3, 4, 5]",
            "[0, 1, 2, 3, 4, 5]",
            "[5]",
        ),
        "A",
    ),
    _mmlu(
        "mmlu-18-prog-http",
        "programming",
        "Which HTTP method is typically used to create a new resource?",
        ("GET", "DELETE", "HEAD", "POST"),
        "D",
    ),
    _mmlu(
        "mmlu-19-prog-git",
        "programming",
        "Which git command creates a new branch and switches to it in one step?",
        ("git branch", "git merge", "git switch -c", "git rebase"),
        "C",
    ),
    _mmlu(
        "mmlu-20-prog-sql",
        "programming",
        "Which SQL clause is used to filter rows after grouping?",
        ("WHERE", "HAVING", "GROUP BY", "ORDER BY"),
        "B",
    ),
]


def _humaneval(
    name: str,
    prompt: str,
    tests: str,
    entry_point: str,
) -> BenchPrompt:
    return BenchPrompt(
        name=name,
        category="humaneval",
        messages=[
            {
                "role": "user",
                "content": (
                    "Complete the following Python function. Respond with a "
                    "single fenced ```python``` code block containing the full "
                    "function definition (you may include the signature and "
                    "docstring). Do not include explanations.\n\n"
                    f"{prompt}"
                ),
            }
        ],
        max_tokens=512,
        grader="code_exec",
        expected={
            "prompt": prompt,
            "tests": tests,
            "entry_point": entry_point,
        },
    )


HUMANEVAL_MINI: list[BenchPrompt] = [
    _humaneval(
        "humaneval-01-add",
        'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n',
        (
            "def check(candidate):\n"
            "    assert candidate(2, 3) == 5\n"
            "    assert candidate(-1, 1) == 0\n"
            "    assert candidate(0, 0) == 0\n"
        ),
        "add",
    ),
    _humaneval(
        "humaneval-02-strlen",
        'def strlen(s: str) -> int:\n    """Return the length of the string s."""\n',
        (
            "def check(candidate):\n"
            "    assert candidate('') == 0\n"
            "    assert candidate('abc') == 3\n"
            "    assert candidate('hello world') == 11\n"
        ),
        "strlen",
    ),
    _humaneval(
        "humaneval-03-flip-case",
        'def flip_case(s: str) -> str:\n    """Flip lowercase to uppercase and vice versa."""\n',
        (
            "def check(candidate):\n"
            "    assert candidate('') == ''\n"
            "    assert candidate('Hello') == 'hELLO'\n"
            "    assert candidate('abcDEF') == 'ABCdef'\n"
        ),
        "flip_case",
    ),
    _humaneval(
        "humaneval-04-get-positive",
        'def get_positive(lst: list) -> list:\n    """Return only the positive numbers from the list, in order."""\n',
        (
            "def check(candidate):\n"
            "    assert candidate([-1, 2, -3, 4]) == [2, 4]\n"
            "    assert candidate([]) == []\n"
            "    assert candidate([0, -1, -2]) == []\n"
            "    assert candidate([1, 2, 3]) == [1, 2, 3]\n"
        ),
        "get_positive",
    ),
    _humaneval(
        "humaneval-05-is-palindrome",
        (
            "def is_palindrome(s: str) -> bool:\n"
            '    """Return True if s reads the same forwards and backwards (case-insensitive, ignoring non-alphanumerics)."""\n'
        ),
        (
            "def check(candidate):\n"
            "    assert candidate('racecar') is True\n"
            "    assert candidate('hello') is False\n"
            "    assert candidate('A man a plan a canal Panama') is True\n"
            "    assert candidate('') is True\n"
        ),
        "is_palindrome",
    ),
    _humaneval(
        "humaneval-06-fibonacci",
        (
            "def fibonacci(n: int) -> int:\n"
            '    """Return the n-th Fibonacci number (fibonacci(0) == 0, fibonacci(1) == 1)."""\n'
        ),
        (
            "def check(candidate):\n"
            "    assert candidate(0) == 0\n"
            "    assert candidate(1) == 1\n"
            "    assert candidate(10) == 55\n"
            "    assert candidate(15) == 610\n"
        ),
        "fibonacci",
    ),
    _humaneval(
        "humaneval-07-digit-sum",
        (
            "def digit_sum(n: int) -> int:\n"
            '    """Return the sum of the decimal digits of n (n >= 0)."""\n'
        ),
        (
            "def check(candidate):\n"
            "    assert candidate(0) == 0\n"
            "    assert candidate(7) == 7\n"
            "    assert candidate(123) == 6\n"
            "    assert candidate(9999) == 36\n"
        ),
        "digit_sum",
    ),
    _humaneval(
        "humaneval-08-largest-divisor",
        (
            "def largest_divisor(n: int) -> int:\n"
            '    """Return the largest proper divisor of n (n > 1)."""\n'
        ),
        (
            "def check(candidate):\n"
            "    assert candidate(15) == 5\n"
            "    assert candidate(100) == 50\n"
            "    assert candidate(7) == 1\n"
        ),
        "largest_divisor",
    ),
    _humaneval(
        "humaneval-09-max-element",
        (
            "def max_element(lst: list) -> int:\n"
            '    """Return the maximum element in a non-empty list of integers."""\n'
        ),
        (
            "def check(candidate):\n"
            "    assert candidate([1, 2, 3]) == 3\n"
            "    assert candidate([-5, -1, -10]) == -1\n"
            "    assert candidate([42]) == 42\n"
        ),
        "max_element",
    ),
    _humaneval(
        "humaneval-10-count-vowels",
        (
            "def count_vowels(s: str) -> int:\n"
            '    """Return the number of vowels (a, e, i, o, u) in s, case-insensitive."""\n'
        ),
        (
            "def check(candidate):\n"
            "    assert candidate('') == 0\n"
            "    assert candidate('hello') == 2\n"
            "    assert candidate('AEIOU') == 5\n"
            "    assert candidate('xyz') == 0\n"
        ),
        "count_vowels",
    ),
]


PROMPT_SETS: dict[str, list[BenchPrompt]] = {
    "gsm8k": GSM8K_MINI,
    "mmlu": MMLU_MINI,
    "humaneval": HUMANEVAL_MINI,
}
