import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from src.core import register


@register
class HybridEncoder(nn.Module):