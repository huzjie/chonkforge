"""Visual grounding benchmark (multimodal: text + image).

Questions reference a synthetic image; the mock backend scores them via the same
deterministic world_answer as text domains, so accuracy tracks `skill`. The CPU
backend actually embeds the image patches, exercising the native-multimodal path.
"""
from .base import BaseBenchmark
from . import register_benchmark
from ..mm.vision import make_image


@register_benchmark("visual_grounding")
class VisualGroundingBenchmark(BaseBenchmark):
    domain = "visual_grounding"
    n_tasks = 12

    def tasks(self):
        qs = [
            "what is in the top-left patch?",
            "which patch is brightest?",
            "how many dark patches?",
            "where is the object?",
            "what color dominates?",
            "which patch is empty?",
            "what shape is in the center?",
            "is the object moving?",
            "what is the spatial relation?",
            "which quadrant has the object?",
            "how many objects?",
            "what is the orientation?",
        ]
        opts = [
            ["an object", "text", "empty", "noise"],
            ["top-right", "bottom-left", "center", "top-left"],
            ["two", "none", "four", "eight"],
            ["center", "off-screen", "left edge", "right edge"],
            ["light", "dark", "red", "green"],
            ["bottom-right", "center", "top-left", "none"],
            ["square", "circle", "triangle", "line"],
            ["static", "fast", "slow", "rotating"],
            ["left of", "above", "inside", "behind"],
            ["Q1", "Q2", "Q3", "Q4"],
            ["one", "two", "three", "zero"],
            ["upright", "tilted", "inverted", "mirrored"],
        ]
        out = []
        for i in range(len(qs)):
            img = make_image(8, 8, key="vg.%d" % i)
            out.append((qs[i], opts[i], img))
        return out
