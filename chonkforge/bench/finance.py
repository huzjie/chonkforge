"""Finance benchmark."""
from .base import BaseBenchmark
from . import register_benchmark


@register_benchmark("finance")
class FinanceBenchmark(BaseBenchmark):
    domain = "finance"
    n_tasks = 16

    def tasks(self):
        qs = [
            "what is compound interest?",
            "what does diversification reduce?",
            "what is a bond?",
            "what is EBITDA?",
            "what is market cap?",
            "what is inflation?",
            "what is a P/E ratio?",
            "what is liquidity?",
            "what is a dividend?",
            "what is risk-adjusted return?",
            "what is hedging?",
            "what is an ETF?",
            "what is net present value?",
            "what is a balance sheet?",
            "what is working capital?",
            "what is a bull market?",
        ]
        opts = [
            ["interest on interest", "flat interest", "a fee", "a tax"],
            ["unsystematic risk", "systematic risk", "all risk", "returns"],
            ["debt instrument", "equity share", "a derivative", "cash"],
            ["earnings metric", "revenue only", "a tax", "a cost"],
            ["shares x price", "total debt", "annual profit", "book value"],
            ["price rise", "price fall", "rate cut", "growth"],
            ["price / earnings", "debt / equity", "assets / sales", "cost / profit"],
            ["ease to convert", "locked up", "illiquid", "a bond"],
            ["profit payout", "a loan", "a fee", "a write-off"],
            ["return per risk", "raw return", "no risk", "a hedge"],
            ["offset risk", "increase risk", "speculate", "leverage"],
            ["basket of assets", "a single stock", "a bond only", "cash"],
            ["discounted cashflows", "sum of costs", "revenue", "a ratio"],
            ["assets = liabilities + equity", "cash only", "profit", "revenue"],
            ["current - liabilities", "fixed assets", "long debt", "revenue"],
            ["rising prices", "falling prices", "flat market", "a crash"],
        ]
        return [(qs[i], opts[i], None) for i in range(len(qs))]
