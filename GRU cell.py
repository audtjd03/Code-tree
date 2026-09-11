# GRU Cell 모델 구현

# 라이브러리 호출
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn import Parameter
from torch.utils.data import DataLoader


# 실행 환경 및 난수 시드 설정
SEED = 125
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
