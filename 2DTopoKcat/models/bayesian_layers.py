import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BayesianLinear(nn.Module):


    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

                                        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))

                                      
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -5.0)
        if self.bias_mu is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, -5.0)

    def forward(self, x):
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)

        return F.linear(x, weight, bias)

    def kl_divergence(self):

        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        kl = self._kl_diag(self.weight_mu, weight_sigma, self.prior_sigma)
        kl += self._kl_diag(self.bias_mu, bias_sigma, self.prior_sigma)
        return kl

    @staticmethod
    def _kl_diag(mu, sigma, prior_sigma):

        sig2 = sigma ** 2
        prior_sig2 = prior_sigma ** 2
        return 0.5 * (sig2 / prior_sig2 + mu ** 2 / prior_sig2 - 1.0 + torch.log(prior_sig2 / sig2)).sum()


class BayesianMLP(nn.Module):


    def __init__(self, dims, prior_sigma=1.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(BayesianLinear(dims[i], dims[i + 1], prior_sigma))
        self.hidden_layers = len(dims) - 2

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        y = x
        for i in range(self.hidden_layers):
            y = F.relu(self.layers[i](y))
        y = self.layers[-1](y)
        return y

    def kl_divergence(self):
        return sum(layer.kl_divergence() for layer in self.layers)
