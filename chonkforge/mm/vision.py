"""Vision patch embedding: images become tokens in the SAME space as text.

The native-multimodal trick: split an image into patches, linearly project each
patch into the transformer's embedding dimension, and hand those vectors to the
*one* backbone as if they were text-token embeddings. No separate frozen vision
encoder + adapter, no cross-attention bridge.
"""
from dataclasses import dataclass, field

from ..utils.stable import stable_vector


@dataclass
class ImageToken:
    """A patch of an image, already projected into the shared embedding space."""
    patch_ids: list = field(default_factory=list)
    vectors: list = field(default_factory=list)  # list[list[float]] dim
    grid: tuple = (0, 0)


class PatchEmbedder:
    """Split a grayscale/color image into patches and project each to `dim`."""

    def __init__(self, dim=32, patch=4, img_h=16, img_w=16, seed=0):
        self.dim = dim
        self.patch = patch
        self.img_h = img_h
        self.img_w = img_w
        self.seed = seed
        # projection weight: patch*patch -> dim
        self.proj = [stable_vector("chonk.proj.%d.%d" % (seed, j), patch * patch, -0.1, 0.1)
                     for j in range(dim)]

    def grid_size(self):
        return (self.img_h // self.patch, self.img_w // self.patch)

    def embed(self, image):
        """image: list[list[float]] (img_h x img_w). Returns ImageToken."""
        gh, gw = self.grid_size()
        vectors = []
        patch_ids = []
        idx = 0
        for gy in range(gh):
            for gx in range(gw):
                vals = []
                for dy in range(self.patch):
                    for dx in range(self.patch):
                        vals.append(image[gy * self.patch + dy][gx * self.patch + dx])
                # linear projection to dim
                vec = [sum(vals[k] * self.proj[j][k] for k in range(len(vals)))
                       for j in range(self.dim)]
                vectors.append(vec)
                patch_ids.append(300 + idx)  # visual token ids start at 300
                idx += 1
        return ImageToken(patch_ids=patch_ids, vectors=vectors, grid=(gh, gw))


def make_image(h, w, key="chonk.img", lo=0.0, hi=1.0):
    """Deterministic synthetic image (list of rows)."""
    vec = stable_vector(key, h * w, lo, hi)
    return [vec[i * w:(i + 1) * w] for i in range(h)]
