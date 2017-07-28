import torch

torch.FloatTensor.cuda()
a = torch.FloatTensor([1])
a.cuda()
torch.autograd.Variable.cuda()
torch.optim.SGD()