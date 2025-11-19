import unittest

import torch

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.linear import LinearBase
from sglang.srt.configs.qwen3_vl import Qwen3VLConfig
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel


class TestEmbedInterpolate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_embed_interpolate(self):
        sarg = ServerArgs(
                   model_path="dummy",
                   device = "npu"
               )
        mconf = Qwen3VLConfig(
                    hidden_size = 1024,
                    num_heads = 16,
                    num_position_embeddings = 2304,
                    patch_size = 16,
                    spatial_merge_size = 2,
                    temporal_patch_size = 2,
                    deepstack_visual_indexes = [5, 11, 17],
                    in_channels = 3,
                    depth = 24,
                    intermediate_size = 4096,
                    hidden_act = "gelu_pytorch_tanh",
                    out_hidden_size = 2560
                )
        set_global_server_args_for_scheduler(sarg)
        init_distributed_environment(
            backend="nccl",
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:2646",
        )
        initialize_model_parallel()
        initialize_dp_attention(
                server_args=sarg,
                model_config=mconf,
            )
        model = Qwen3VLMoeVisionModel(
            mconf,
            quant_config=None,
            norm_eps=1e-6,
            prefix="visual",
        )
        embeddings = model.fast_pos_embed_interpolate([(16,256,256),(32,512,512)])

        embeddings_s0 = embeddings[:256*256, :]
        embeddings_s1 = embeddings[256*256:2*256*256, :]
        self.assertTrue(torch.allclose(embeddings_s0, embeddings_s1, atol=5e-3))

        embeddings_l = embeddings[16*256*256:16*256*256+512*512, :]
        embeddings_r = (embeddings_l[::4, :]).reshape(128,2,128,2,-1).permute(0,2,1,3,4).reshape(256*256, -1)
        print((embeddings_s0[:, 0])[:12])
        print((embeddings_r[:, 0])[:12])
        self.assertTrue(torch.allclose(embeddings_s0, embeddings_r, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
