'''
getting four side of mango
specially two big side mango

'''
#
import torch
import torch.nn as nn
import torch.nn.functional as F


class netBlock(nn.Module):

    def __init__(self):
        super(netBlock, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 10)  # 5*5 from image dimension
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 10)
        self.fc7 = nn.Linear(10, 10)
        self.fc8 = nn.Linear(10, 10)
        self.fc9 = nn.Linear(10, 10)
        self.fc10 = nn.Linear(10, 10)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))

        x = self.fc10(x)
        return x
#
#
# net = netBlock()
# print(netBlock)
#
# params = list(net.parameters())
# print(len(params))
# print(params[0].size())  # conv1's .weight
#
# input = torch.randn(1, 1, 32, 32)
# out = netBlock(input)
# print(out)
#
# netBlock.zero_grad()
# out.backward(torch.randn(1, 10))
#
# output = netBlock(input)
# target = torch.randn(10)  # a dummy target, for example
# target = target.view(1, -1)  # make it the same shape as output
# criterion = nn.MSELoss()
#
# loss = criterion(output, target)
# print(loss)
#
# print(loss.grad_fn)  # MSELoss
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
#
# netBlock.zero_grad()     # zeroes the gradient buffers of all parameters
#
# print('conv1.bias.grad before backward')
# print(netBlock.conv1.bias.grad)
#
# loss.backward()
#
# print('conv1.bias.grad after backward')
# print(netBlock.conv1.bias.grad)
#
# learning_rate = 0.01
# for f in netBlock.parameters():
#     f.data.sub_(f.grad.data * learning_rate)
#
# import torch.optim as optim
#
# # create your optimizer
# optimizer = optim.SGD(netBlock.parameters(), lr=0.01)
#
# # in your training loop:
# optimizer.zero_grad()   # zero the gradient buffers
# output = netBlock(input)
# loss = criterion(output, target)
# loss.backward()
# optimizer.step()    # Does the update



#neural Encoder


class encoderNetBlock(nn.Module):

    def __init__(self):
        super(encoderNetBlock, self).__init__()

        self.fc1 = nn.Linear(10, 8)  # 5*5 from image dimension
        self.fc2 = nn.Linear(8, 6)
        self.fc3 = nn.Linear(6, 4)
        self.fc4 = nn.Linear(4, 2)
        self.fc5 = nn.Linear(2, 1)


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = self.fc5(x)
        return x


net = encoderNetBlock()
print(encoderNetBlock)

# params = list(net.parameters())
# print(len(params))
# print(params[0].size())  # conv1's .weight
#
# input = torch.randn(10,10)
# out = encoderNetBlock(input)
# print(out)
#
# encoderNetBlock.zero_grad()
# out.backward(torch.randn(1, 10))
#
# output = encoderNetBlock(input)
# target = torch.randn(1)  # a dummy target, for example
# target = target.view(1, -1)  # make it the same shape as output
# criterion = nn.MSELoss()
#
# loss = criterion(output, target)
# print(loss)
#
# print(loss.grad_fn)  # MSELoss
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
#
# encoderNetBlock.zero_grad()     # zeroes the gradient buffers of all parameters
#
#
#
#
#
# learning_rate = 0.01
# for f in encoderNetBlock.parameters():
#     f.data.sub_(f.grad.data * learning_rate)
#
# import torch.optim as optim
#
# # create your optimizer
# optimizer = optim.SGD(encoderNetBlock.parameters(), lr=0.01)
#
# # in your training loop:
# optimizer.zero_grad()   # zero the gradient buffers
# output = encoderNetBlock(input)
# loss = criterion(output, target)
# loss.backward()
# optimizer.step()    # Does the update
#


class decoderNetBlock(nn.Module):

    def __init__(self):
        super(decoderNetBlock, self).__init__()

        self.fc1 = nn.Linear(1, 2)  # 5*5 from image dimension
        self.fc2 = nn.Linear(2, 4)
        self.fc3 = nn.Linear(4, 6)
        self.fc4 = nn.Linear(6, 8)
        self.fc5 = nn.Linear(9, 10)


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = self.fc5(x)
        return x


net = decoderNetBlock()
print(decoderNetBlock)


#master parameter z
netBlock
encoderNetBlock

#---feature space
#teacher y
netBlock
encoderNetBlock

#student x
netBlock
encoderNetBlock

#unsupervisied deta
netBlock
encoderNetBlock

#reinforcement phi
netBlock
decoderNetBlock

