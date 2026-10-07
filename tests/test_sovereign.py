"""Unit tests. Run from repo root:  python -m unittest discover -s tests -t ."""
import unittest

from chonkforge.sovereign import SovereignCompute, DataGovernance


class TestSovereign(unittest.TestCase):
    def test_region_check(self):
        sc = SovereignCompute(region="eu", allowed_regions=["eu"])
        ok, _ = sc.check("eu")
        self.assertTrue(ok)
        ok2, _ = sc.check("us")
        self.assertFalse(ok2)

    def test_governance_audit(self):
        sc = SovereignCompute(region="eu", allowed_regions=["eu"])
        gov = DataGovernance(sc)
        gov.access("alice", "eu", "inference")
        gov.export("bob", "us", "finetune")
        self.assertEqual(len(gov.log.to_list()), 2)


if __name__ == "__main__":
    unittest.main()
