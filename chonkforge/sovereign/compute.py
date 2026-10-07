"""Sovereign compute abstraction.

Reusable idea behind "sovereign AI": pin model weights, data, and compute to a
specific region / jurisdiction so regulated orgs keep control over residency and
continuity. This is a lightweight policy layer -- it does not move bytes, it
*asserts and audits* where bytes are allowed to live.
"""
from dataclasses import dataclass, field


@dataclass
class SovereignCompute:
    region: str = "eu"
    jurisdiction: str = "GDPR"
    data_residency: str = "eu-west"
    allowed_regions: list = field(default_factory=lambda: ["eu"])

    def check(self, request_region):
        """Returns (ok, reason) for whether a request region is admissible."""
        if request_region not in self.allowed_regions:
            return False, "region %r not in allowed %r" % (request_region, self.allowed_regions)
        return True, "ok"

    def describe(self):
        return {
            "region": self.region,
            "jurisdiction": self.jurisdiction,
            "data_residency": self.data_residency,
            "allowed_regions": self.allowed_regions,
        }
