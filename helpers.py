import torch


def logsumexp(x, dim):
    """Logsumexp trick to avoid overflow in a log of sum of exponential expression

    Args:
        x (Variable or Tensor): the input on which to compute the log of sum of exponential

    Returns:
        logsum (Variable or Tensor): the computed log of sum of exponential
    """

    assert x.dim() == 2
    x_max, x_max_idx = x.max(dim=dim, keepdim=True)
    logsum = x_max + torch.log((x - x_max).exp().sum(dim=dim, keepdim=True))
    return logsum


def compute_KL(x, mu, sigma, prior):
    """
    Compute KL divergence between posterior and prior.
    """

    posterior = torch.distributions.Normal(mu, sigma)
    log_posterior = posterior.log_prob(x).sum()

    device = x.device

    N1 = torch.distributions.Normal(
        torch.FloatTensor(1).fill_(0.0).to(device),
        torch.FloatTensor(1).fill_(prior.sigma1).to(device),
    )
    N2 = torch.distributions.Normal(
        torch.FloatTensor(1).fill_(0.0).to(device),
        torch.FloatTensor(1).fill_(prior.sigma2).to(device),
    )

    prior1 = prior.pi_mixture * N1.log_prob(x).exp()
    prior2 = (1.0 - prior.pi_mixture) * N2.log_prob(x).exp()

    prior = prior1 + prior2
    log_prior = prior.log().sum()

    return log_posterior - log_prior
