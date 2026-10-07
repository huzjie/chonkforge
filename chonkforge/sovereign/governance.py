"""Data governance with an append-only audit log."""
from dataclasses import dataclass, field


@dataclass
class AuditLog:
    entries: list = field(default_factory=list)

    def record(self, event, actor, detail=""):
        self.entries.append({"event": event, "actor": actor, "detail": detail})

    def to_list(self):
        return list(self.entries)


class DataGovernance:
    def __init__(self, compute=None):
        from .compute import SovereignCompute
        self.compute = compute or SovereignCompute()
        self.log = AuditLog()

    def access(self, actor, region, purpose):
        ok, reason = self.compute.check(region)
        self.log.record("access", actor, "region=%s purpose=%s -> %s" % (region, purpose, reason))
        return ok

    def export(self, actor, region, dataset):
        ok, reason = self.compute.check(region)
        self.log.record("export", actor, "dataset=%s -> %s" % (dataset, reason))
        return ok
