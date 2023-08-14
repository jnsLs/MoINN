from torch.nn import functional


def softmax(x):
    return functional.softmax(x, dim=-1)


def swish(x):
    return x * functional.sigmoid(x)
