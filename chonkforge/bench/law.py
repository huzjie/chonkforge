"""Law benchmark."""
from .base import BaseBenchmark
from . import register_benchmark


@register_benchmark("law")
class LawBenchmark(BaseBenchmark):
    domain = "law"
    n_tasks = 16

    def tasks(self):
        qs = [
            "what is a contract?",
            "what is intellectual property?",
            "what is 'precedent'?",
            "what is a tort?",
            "what is jurisdiction?",
            "what is GDPR about?",
            "what is due diligence?",
            "what is a non-disclosure agreement?",
            "what is arbitration?",
            "what is liability?",
            "what is a fiduciary duty?",
            "what is 'force majeure'?",
            "what is a patent?",
            "what is a trademark?",
            "what is compliance?",
            "what is a clause?",
        ]
        opts = [
            ["legally binding agreement", "a suggestion", "an email", "a memo"],
            ["creations of the mind", "real estate", "cash", "stocks"],
            ["prior ruling guides", "new law", "a statute", "a contract"],
            ["civil wrong", "a crime", "a contract", "a right"],
            ["court's authority", "a building", "a judge", "a case"],
            ["data protection", "tax", "antitrust", "labor"],
            ["investigation before deal", "after deal", "no review", "a tax"],
            ["confidentiality agreement", "a loan", "a sale", "a lease"],
            ["private dispute resolution", "public trial", "appeal", "mediation"],
            ["legal responsibility", "a right", "immunity", "a privilege"],
            ["act in best interest", "act selfishly", "no duty", "a contract"],
            ["unforeseeable event", "a planned event", "a breach", "a payment"],
            ["invention right", "a logo", "a brand", "a trade secret"],
            ["brand identifier", "an invention", "a process", "a formula"],
            ["follow rules", "break rules", "ignore", "no standard"],
            ["contract provision", "a lawsuit", "a judge", "a court"],
        ]
        return [(qs[i], opts[i], None) for i in range(len(qs))]
