"""
KL-annealed ELBO for Bayesian SVI training.

Subclasses TraceMeanField_ELBO to decompose the ELBO into NLL and KL terms,
applying a tunable weight to the KL term. This allows KL annealing: ramping
the KL weight from a small value to 1.0 over the first few epochs so the
model can learn the data distribution before prior regularization kicks in.

The kl_weight attribute is set externally by KLAnnealingCallback each epoch.
"""

import torch
from torch.distributions import kl_divergence

from pyro.distributions.util import scale_and_mask
from pyro.infer.trace_mean_field_elbo import TraceMeanField_ELBO
from pyro.infer.util import is_validation_enabled, check_fully_reparametrized
from pyro.util import warn_if_nan


class KLAnnealedELBO(TraceMeanField_ELBO):
    """TraceMeanField_ELBO with a tunable KL weight for annealing.

    When kl_weight=1.0, this is mathematically equivalent to TraceMeanField_ELBO.
    When kl_weight<1.0, the KL divergence term is down-weighted, reducing prior
    pressure during early training.

    Args:
        kl_weight: Multiplicative weight for the KL divergence term (0 to 1).
        **kwargs: Passed to TraceMeanField_ELBO (e.g., num_particles).
    """

    def __init__(self, kl_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.kl_weight = kl_weight

    def differentiable_loss(self, model, guide, *args, **kwargs):
        """Compute KL-annealed ELBO loss.

        Note: TraceMeanField_ELBO requires fully reparameterized distributions,
        so the surrogate-loss trick from Trace_ELBO is unnecessary (surrogate = loss).
        """
        nll, kl, total = self._compute_parts(model, guide, *args, **kwargs)
        return total

    def differentiable_loss_with_parts(self, model, guide, *args, **kwargs):
        """Return (nll, weighted_kl, total) for logging.

        Returns:
            Tuple of (nll, kl_weight * kl, nll + kl_weight * kl).
            All three are differentiable tensors.
        """
        return self._compute_parts(model, guide, *args, **kwargs)

    def _compute_parts(self, model, guide, *args, **kwargs):
        """Decompose ELBO into NLL and KL, apply kl_weight to KL.

        Iterates over model trace sites:
        - Observed sites contribute to NLL (negative log-likelihood).
        - Latent sites contribute to KL (KL divergence from guide to prior).

        Uses analytic KL divergence when available (same as TraceMeanField_ELBO),
        with fallback to sampling-based KL when kl_divergence is not implemented.

        Returns:
            Tuple of (nll, kl_weight * kl, nll + kl_weight * kl).
        """
        # Accumulators start as Python 0.0 — torch's operator dispatch handles
        # device placement when the first tensor is added, avoiding the
        # CPU-tensor-on-GPU-data pitfall.
        nll_total = 0.0
        kl_total = 0.0

        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            nll_particle = 0.0
            kl_particle = 0.0

            for name, model_site in model_trace.nodes.items():
                if model_site["type"] != "sample":
                    continue

                if model_site["is_observed"]:
                    # Observed site: contributes to NLL
                    # log_prob_sum is log p(y|x,theta), negate for NLL
                    nll_particle = nll_particle - model_site["log_prob_sum"]
                else:
                    # Latent site: contributes to KL
                    guide_site = guide_trace.nodes[name]
                    if is_validation_enabled():
                        check_fully_reparametrized(guide_site)

                    try:
                        kl_qp = kl_divergence(guide_site["fn"], model_site["fn"])
                        kl_qp = scale_and_mask(
                            kl_qp, scale=guide_site["scale"], mask=guide_site["mask"]
                        )
                        if torch.is_tensor(kl_qp):
                            assert (
                                torch._C._get_tracing_state()
                                or kl_qp.shape == guide_site["fn"].batch_shape
                            )
                            kl_qp_sum = kl_qp.sum()
                        else:
                            kl_qp_sum = (
                                kl_qp * torch.Size(guide_site["fn"].batch_shape).numel()
                            )
                        kl_particle = kl_particle + kl_qp_sum
                    except NotImplementedError:
                        # Fallback: guide_log_prob - model_log_prob
                        entropy_term = guide_site["score_parts"].entropy_term
                        kl_particle = (
                            kl_particle
                            + entropy_term.sum()
                            - model_site["log_prob_sum"]
                        )

            # Handle auxiliary sites in the guide (same as parent)
            for name, guide_site in guide_trace.nodes.items():
                if guide_site["type"] == "sample" and name not in model_trace.nodes:
                    assert guide_site["infer"].get("is_auxiliary")
                    if is_validation_enabled():
                        check_fully_reparametrized(guide_site)
                    entropy_term = guide_site["score_parts"].entropy_term
                    kl_particle = kl_particle + entropy_term.sum()

            nll_total = nll_total + nll_particle / self.num_particles
            kl_total = kl_total + kl_particle / self.num_particles

        weighted_kl = self.kl_weight * kl_total
        total = nll_total + weighted_kl

        warn_if_nan(total, "loss")
        return nll_total, weighted_kl, total
