"""Coding benchmark."""
from .base import BaseBenchmark
from . import register_benchmark


@register_benchmark("coding")
class CodingBenchmark(BaseBenchmark):
    domain = "coding"
    n_tasks = 16

    def tasks(self):
        qs = [
            "which data structure gives O(1) average lookup?",
            "what does 'git rebase' do?",
            "which sort is O(n log n) worst-case?",
            "what is a closure?",
            "which HTTP verb is idempotent?",
            "what is the time complexity of binary search?",
            "what is a deadlock?",
            "which keyword declares a constant in Rust?",
            "what does a decorator do in Python?",
            "which is a NoSQL database?",
            "what is memoization?",
            "which language is statically typed?",
            "what is a linked list?",
            "what does 'idempotent' mean?",
            "which is a version control system?",
            "what is tail recursion?",
        ]
        opts = [
            ["hash table", "linked list", "stack", "queue"],
            ["rewrite history", "stage files", "fetch remote", "clone repo"],
            ["merge sort", "bubble sort", "insertion sort", "selection sort"],
            ["function + captured scope", "a loop", "a class", "a module"],
            ["PUT", "POST", "PATCH", "DELETE"],
            ["O(log n)", "O(n)", "O(n log n)", "O(1)"],
            ["circular resource wait", "a race", "a segfault", "a timeout"],
            ["const", "let", "var", "auto"],
            ["wraps a function", "defines a loop", "imports a module", "raises errors"],
            ["MongoDB", "PostgreSQL", "MySQL", "SQLite"],
            ["cache results", "sort data", "encrypt data", "log data"],
            ["Rust", "Python", "JavaScript", "Ruby"],
            ["nodes + pointers", "contiguous array", "key-value", "tree only"],
            ["same result on repeat", "fails on repeat", "random result", "slower on repeat"],
            ["git", "npm", "pip", "cargo only"],
            ["recursion as last call", "recursion + loop", "no recursion", "stack overflow"],
        ]
        return [(qs[i], opts[i], None) for i in range(len(qs))]
