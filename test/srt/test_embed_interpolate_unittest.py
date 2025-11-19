import unittest

import torch

from sglang.srt.mem_cache.allocator import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache


class TestSWA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_embed_interpolate(self):
        size = 16
        size_swa = 16
        head_num = 8
        head_dim = 128
        num_layers = 48
        global_interval = 4
        dtype = torch.bfloat16
        device = "cuda"
        full_attention_layer_ids = [i for i in range(0, num_layers, global_interval)]
        full_attention_layer_ids_set = set(full_attention_layer_ids)
        swa_attention_layer_ids = [
            i for i in range(num_layers) if i not in full_attention_layer_ids_set
        ]

    def __init__(
        self,
        config: Qwen3VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3LLMModel,
    ) -> None:
        super().__init__()

        model = Qwen3VLMoeVisionModel(
            Qwen3VLConfig(),
            quant_config=QuantizationConfig(),
            norm_eps=1e-6,
            prefix="visual",
        )
        embeddings = model.fast_pos_embed_interpolate([(16,256,256),(32,512,512)]):
        embeddings_s = embeddings[:16*256*256]
        embeddings_l = embeddings[16*256*256:]
        embeddings_r = embeddings_l[::2, ::2, ::2]
        self.assertEqual(embeddings_s, embeddings_r)


if __name__ == "__main__":
    unittest.main()
