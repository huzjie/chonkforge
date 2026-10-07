"""Cybersecurity benchmark."""
from .base import BaseBenchmark
from . import register_benchmark


@register_benchmark("cybersecurity")
class CybersecurityBenchmark(BaseBenchmark):
    domain = "cybersecurity"
    n_tasks = 16

    def tasks(self):
        qs = [
            "what is SQL injection?",
            "best defense against XSS?",
            "what is a zero-day?",
            "what does 'least privilege' mean?",
            "what is phishing?",
            "what is MFA?",
            "what is a sandbox escape?",
            "what is end-to-end encryption?",
            "what does a WAF do?",
            "what is a supply-chain attack?",
            "what is key rotation?",
            "what is RBAC?",
            "what is a honeypot?",
            "what is RCE?",
            "what is credential stuffing?",
            "what is an audit log for?",
        ]
        opts = [
            ["inject SQL via input", "a database", "a query cache", "an index"],
            ["output encoding", "bigger server", "more RAM", "open ports"],
            ["unknown vuln", "known patch", "a type of malware", "a firewall"],
            ["minimal access", "full admin", "no access", "shared root"],
            ["deceptive message", "a firewall", "a hash", "a backup"],
            ["multi-factor auth", "one password", "a token only", "a cookie"],
            ["escape isolation", "stay inside", "close a sandbox", "log out"],
            ["only endpoints read", "server reads", "no encryption", "plaintext"],
            ["filter web traffic", "compile code", "store data", "run DB"],
            ["attack via dependency", "direct attack", "a DDoS", "a worm"],
            ["periodic key change", "never change", "one key", "no keys"],
            ["role-based access", "no roles", "flat perms", "no auth"],
            ["decoy system", "real DB", "a firewall", "a patch"],
            ["remote code execution", "a crash", "a log", "a backup"],
            ["reuse stolen creds", "new creds", "a virus", "a scan"],
            ["forensics + compliance", "nothing", "speed", "marketing"],
        ]
        return [(qs[i], opts[i], None) for i in range(len(qs))]
