# from reconstruct.model import ConvNormLayer, FrozenBatchNorm2d, get_activation
# import torch.nn
# import torch
# import sys,os
# sys.append(".")
# print(os.getcwd())

# class Test_common:
#     def test_ConvNormLayer(self):
#         x = torch.ones([3, 8, 800, 800])
#         model = ConvNormLayer(
#             ch_in=8,
#             ch_out=8,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             bias=True,
#             act='silu'
#         )
#         assert x.shape == model(x).shape

# # D_MODEL = 6
# # NUM_HEAD = 2
# # class Testbselfattention:
# #     def test_selfattention(self):
# #         # batch,rows,heads,d_tensor
# #         x = torch.ones([1,3,2,3])
# #         model = selfattention()

# #         assert model(x,x,x).shape == x.shape
