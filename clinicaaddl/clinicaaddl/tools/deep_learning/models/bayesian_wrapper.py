import torch.nn as nn
import torch

class BayesianWrapper(nn.Module):
    APPLY_TO = [nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.ConvTranspose3d]

    class BayesianDropout(nn.Module):
        def __init__(self, p=0.2, inplace=False):
            super(BayesianWrapper.BayesianDropout, self).__init__()
            self.dropout = nn.Dropout(p=p, inplace=inplace)

        def forward(self, input):
            return self.dropout.forward(input)

        def train(self, mode=True):
            """
            train/eval settings have no effect on bayesian dropouts
            :param mode:
            :return:
            """
            return self

    class BayesianSequential(nn.Sequential):
        def __init__(self, layer):
            super(BayesianWrapper.BayesianSequential, self).__init__(BayesianWrapper.BayesianDropout(),
                                                                     layer)

    def __init__(self, model):
        """
        :param model:
        :type model: nn.Model
        """
        super(BayesianWrapper, self).__init__()
        self.model = model
        self.model.apply(BayesianWrapper.make_bayesian)
        self.is_bayesian = True

    def make_bayesian(self):
        for key in self._modules:
            if type(self._modules[key]) in BayesianWrapper.APPLY_TO:
                self._modules[key] = BayesianWrapper.BayesianSequential(self._modules[key])

    def remove_bayesian(self):
        for key in self._modules:
            if type(self._modules[key]) is BayesianWrapper.BayesianSequential:
                self._modules[key] = self._modules[key]._modules["1"]

    def bayesian(self, mode=True):
        if self.is_bayesian and not mode:
            self.model.apply(BayesianWrapper.remove_bayesian)
        if not self.is_bayesian and mode:
            self.model.apply(BayesianWrapper.make_bayesian)
        self.is_bayesian = mode

    def load_state_dict(self, state_dict, strict=True):
        self.model.apply(BayesianWrapper.remove_bayesian)
        self.model.load_state_dict(state_dict, strict=strict)
        if self.is_bayesian:
            self.model.apply(BayesianWrapper.make_bayesian)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        self.model.apply(BayesianWrapper.remove_bayesian)
        state_dict = self.model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if self.is_bayesian:
            self.model.apply(BayesianWrapper.make_bayesian)
        return state_dict

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def save(self, model_file):
        torch.save(self.state_dict(), model_file)

    def forward(self, *input,**kwargs):
        #return self.model.forward(*input, kwargs) #-> gives an error 'forward() takes 2 positional arguments but 3 were given'

        return self.model.forward(*input, **kwargs)

    def __repr__(self):
        return self.model.__repr__()
