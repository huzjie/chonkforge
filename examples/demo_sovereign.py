"""chonkforge example.

Run from anywhere:  python examples/demo_sovereign.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chonkforge.sovereign import SovereignCompute, DataGovernance


def main():
    sc = SovereignCompute(region="eu", jurisdiction="GDPR",
                          data_residency="eu-west", allowed_regions=["eu"])
    gov = DataGovernance(sc)
    print("access from eu :", gov.access("alice", "eu", "inference"))
    print("access from us :", gov.access("alice", "us", "inference"))
    print("export from us :", gov.export("bob", "us", "finetune-v1"))
    for e in gov.log.to_list():
        print("audit:", e)


if __name__ == "__main__":
    main()
