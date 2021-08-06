def freeze_net(net):
    for param in net.parameters():
        param.requires_grad = False
    net.eval()
    return net