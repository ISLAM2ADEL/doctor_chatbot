import os
import torch

# مفاتيح الوصول
HF_TOKEN = "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"
NGROK_AUTH_TOKEN = "2wgvwM4oxNq8BKoqLh4YyywaGex_3fMrAAx1W3MYEEtyYcgKY"

# إعدادات الجهاز (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
