import torch
from collections import defaultdict, Counter, OrderedDict

class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

def accuracy(predictions, targets):

    predicted_classes = predictions.topk(1)[1]
    acc = torch.eq(predicted_classes.squeeze(1), targets).sum().item()/targets.size(0)

    return acc

def two_class_accuracy(predictions, targets):

    m0 = targets == 0
    m1 = targets == 1

    predicted_classes = predictions.topk(1)[1]

    acc = torch.eq(predicted_classes.squeeze(1), targets).sum().item()/targets.size(0)
    acc0 = torch.eq(predicted_classes.squeeze(1).masked_select(m0), targets.masked_select(m0)).sum().item()/targets.masked_select(m0).size(0)
    acc1 = torch.eq(predicted_classes.squeeze(1).masked_select(m1), targets.masked_select(m1)).sum().item()/targets.masked_select(m1).size(0)

    return acc, acc0, acc1
