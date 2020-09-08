import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import tqdm
from pyro import poutine
from pyro.infer import SVI, TraceEnum_ELBO, Predictive, infer_discrete
from pyro.infer.autoguide import init_to_sample, AutoDelta
from pyro.ops.indexing import Vindex
from pyro.optim import Adam
from pyro.util import ignore_jit_warnings
from torch.distributions import AffineTransform
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from multiple_formatter import Multiple
from protein_parser import ProteinParser


def torus_dbn(phis=None, psis=None, lengths=None,
              num_sequences=None, num_states=55,
              prior_conc=0.1, prior_loc=0.0,
              prior_length_shape=100., prior_length_rate=100.,
              prior_kappa_min=10., prior_kappa_max=1000.):
    # From https://pyro.ai/examples/hmm.html
    with ignore_jit_warnings():
        if lengths is not None:
            assert num_sequences is None
            num_sequences = int(lengths.shape[0])
        else:
            assert num_sequences is not None
    transition_probs = pyro.sample('transition_probs',
                                   dist.Dirichlet(torch.ones(num_states, num_states, dtype=torch.float)
                                                  * num_states)
                                   .to_event(1))
    length_shape = pyro.sample('length_shape', dist.HalfCauchy(prior_length_shape))
    length_rate = pyro.sample('length_rate', dist.HalfCauchy(prior_length_rate))
    phi_locs = pyro.sample('phi_locs',
                           dist.VonMises(torch.ones(num_states, dtype=torch.float) * prior_loc,
                                         torch.ones(num_states, dtype=torch.float) * prior_conc).to_event(1))
    phi_kappas = pyro.sample('phi_kappas', dist.Uniform(torch.ones(num_states, dtype=torch.float) * prior_kappa_min,
                                                        torch.ones(num_states, dtype=torch.float) * prior_kappa_max
                                                        ).to_event(1))
    psi_locs = pyro.sample('psi_locs',
                           dist.VonMises(torch.ones(num_states, dtype=torch.float) * prior_loc,
                                         torch.ones(num_states, dtype=torch.float) * prior_conc).to_event(1))
    psi_kappas = pyro.sample('psi_kappas', dist.Uniform(torch.ones(num_states, dtype=torch.float) * prior_kappa_min,
                                                        torch.ones(num_states, dtype=torch.float) * prior_kappa_max
                                                        ).to_event(1))
    element_plate = pyro.plate('elements', 1, dim=-1)
    with pyro.plate('sequences', num_sequences, dim=-2) as batch:
        if lengths is not None:
            lengths = lengths[batch]
            obs_length = lengths.float().unsqueeze(-1)
        else:
            obs_length = None
        state = 0
        sam_lengths = pyro.sample('length',
                                  dist.TransformedDistribution(
                                      dist.GammaPoisson(length_shape, length_rate),
                                      AffineTransform(0., 1.)),
                                  obs=obs_length)
        if lengths is None:
            lengths = sam_lengths.squeeze(-1).long()
        for t in pyro.markov(range(lengths.max())):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                state = pyro.sample(f'state_{t}', dist.Categorical(transition_probs[state]),
                                    infer={'enumerate': 'parallel'})
                if phis is not None:
                    obs_phi = Vindex(phis)[batch, t].unsqueeze(-1)
                else:
                    obs_phi = None
                if psis is not None:
                    obs_psi = Vindex(psis)[batch, t].unsqueeze(-1)
                else:
                    obs_psi = None
                with element_plate:
                    pyro.sample(f'phi_{t}', dist.VonMises(phi_locs[state], phi_kappas[state]), obs=obs_phi)
                    pyro.sample(f'psi_{t}', dist.VonMises(psi_locs[state], psi_kappas[state]), obs=obs_psi)


def main(_argv):
    aas, ds, phis, psis, lengths = ProteinParser.parsef_tensor('../data/torus_dbn/top500.txt')
    guide = AutoDelta(poutine.block(torus_dbn, hide_fn=lambda site: site['name'].startswith('state')),
                      init_loc_fn=init_to_sample)
    svi = SVI(torus_dbn, guide, Adam(dict(lr=0.1)), TraceEnum_ELBO())
    plot_rama(lengths, phis, psis, filename='ground_truth')
    total_iters = 100
    num_states = 55
    plot_rate = 5
    dataset = TensorDataset(phis, psis, lengths)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_losses = []
    with tqdm.trange(total_iters) as pbar:
        total_loss = float('inf')
        for i in pbar:
            losses = []
            num_batches = 0
            for j, (phis, psis, lengths) in enumerate(dataloader):
                loss = svi.step(phis, psis, lengths, num_states=num_states)
                losses.append(loss)
                num_batches += 1
                pbar.set_description_str(f"SVI (batch {j}/{len(dataset)//batch_size}):"
                                         f" {loss / batch_size:.4} [epoch loss: {total_loss:.4}]",
                                         refresh=True)
            total_loss = np.sum(losses) / (batch_size * num_batches)
            total_losses.append(total_loss)
            pbar.set_description_str(f"SVI (batch {j}/{len(dataset)//batch_size}):"
                                     f" {loss / batch_size:.4} [epoch loss: {total_loss:.4}]",
                                     refresh=True)
            if i % plot_rate == 0:
                sample_and_plot(torus_dbn, guide, filename=f'learned_{i}',
                                num_sequences=len(dataset), num_states=num_states)
    sample_and_plot(torus_dbn, guide, filename=f'learned_finish',
                    num_sequences=len(dataset), num_states=num_states)
    plot_losses(total_losses)


def sample_and_plot(model, guide, filename=None, num_sequences=128, num_states=5):
    guide_trace = poutine.trace(guide).get_trace(num_sequences=num_sequences, num_states=num_states)
    samples = infer_discrete(poutine.trace(poutine.replay(model, guide_trace)).get_trace,
                             first_available_dim=-3)(num_sequences=num_sequences,
                                                     num_states=num_states)
    lengths = samples.nodes['length']['value'].squeeze(-1).int()
    phis = torch.cat([site['value'] for name, site in samples.nodes.items()
                      if re.match(r'phi_\d', name)], dim=-1)
    psis = torch.cat([site['value'] for name, site in samples.nodes.items()
                      if re.match(r'psi_\d', name)], dim=-1)
    plot_rama(lengths, phis, psis, filename=filename)


def plot_rama(lengths, phis, psis, filename='rama', dir='figs'):
    fig, ax = plt.subplots()
    ax.hexbin(np.concatenate([phiseq[:lengths[t]].flatten() for t, phiseq in enumerate(phis.detach().numpy())]),
              np.concatenate([psiseq[:lengths[t]].flatten() for t, psiseq in enumerate(psis.detach().numpy())]),
              bins='log')
    multiple = Multiple()
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\psi$')
    ax.xaxis.set_major_formatter(multiple.formatter())
    ax.xaxis.set_major_locator(multiple.locator())
    ax.yaxis.set_major_formatter(multiple.formatter())
    ax.yaxis.set_major_locator(multiple.locator())
    os.makedirs(dir, exist_ok=True)
    fig.savefig(os.path.join(dir, f'{filename}.png'))
    plt.close(fig)


def plot_losses(total_losses, filename='elbo', dir='figs'):
    fig, ax = plt.subplots()
    ax.plot(total_losses)
    os.makedirs(dir, exist_ok=True)
    fig.savefig(os.path.join(dir, f'{filename}.png'))
    plt.close(fig)


if __name__ == '__main__':
    main(sys.argv)
