"""Agentic workflow benchmark."""
from .base import BaseBenchmark
from . import register_benchmark


@register_benchmark("agentic")
class AgenticBenchmark(BaseBenchmark):
    domain = "agentic"
    n_tasks = 16

    def tasks(self):
        qs = [
            "agent should break a task into?",
            "what does a tool-call loop need?",
            "best way to guard agent actions?",
            "what is a rollout?",
            "which signal helps agents recover from failure?",
            "what is an agent 'memory'?",
            "what is the first step in planning?",
            "which is a multi-agent pattern?",
            "what does 'replan' mean?",
            "which makes an agent 'autonomous'?",
            "what is a reward signal?",
            "what is a sandbox for?",
            "what does tool-use verification do?",
            "which is a guardrail?",
            "what is delegation?",
            "what does a state machine give an agent?",
        ]
        opts = [
            ["subtasks", "one big step", "no plan", "random"],
            ["stop condition", "no loop", "only one tool", "no tools"],
            ["permission gates", "no gates", "root access", "open network"],
            ["one execution", "a file", "a model", "a dataset"],
            ["error feedback", "ignore errors", "retry blind", "halt always"],
            ["stored context", "a cache", "a log only", "nothing"],
            ["define goal", "write code", "deploy", "celebrate"],
            ["manager/worker", "single agent", "no agents", "batch job"],
            ["re-plan on new info", "never change", "give up", "loop forever"],
            ["acts without prompts", "needs every step", "read-only", "no tools"],
            ["scalar feedback", "a file", "a token", "a string only"],
            ["safe isolation", "production", "no isolation", "faster"],
            ["check tool output", "skip check", "trust blindly", "ignore"],
            ["output validation", "no validation", "open egress", "no logs"],
            ["hand off subtask", "do everything", "block others", "duplicate"],
            ["explicit states", "no states", "chaos", "random jumps"],
        ]
        return [(qs[i], opts[i], None) for i in range(len(qs))]
