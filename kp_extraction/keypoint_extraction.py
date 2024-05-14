import ultralytics 
import torch

ultralytics.checks()


x = torch.rand(2,2)

print(x)


print(torch.cuda.is_available())