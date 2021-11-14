from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.functional import jacobian
from torch.utils.data import TensorDataset, DataLoader

from src.nn.MDN import MDN, mdn_loss
from src.nn.MLP import MLP


class EnsembleRegressor(nn.Module):

    def __init__(self,
                 num_learners: int,
                 learner_config: dict):
        # To compute the predictive mean and standard deviation of the ensemble model,
        # follow the original derivation of deep ensemble paper.
        assert learner_config['mdn_config'][
                   'num_gaussians'] == 1, "EnsembleRegressor supports only one Gaussian output cases."

        super(EnsembleRegressor, self).__init__()
        self.num_learners = num_learners

        # setup the base learners and their optimizers
        self.learners = nn.ModuleList()
        self.optimizers = []
        for i in range(num_learners):
            learner = nn.Sequential(
                MLP(**learner_config['mlp_config']),
                MDN(**learner_config['mdn_config'])
            )
            self.learners.append(learner)

            opt_config = deepcopy(learner_config['opt_config'])
            opt_name = opt_config.pop('name')
            opt = getattr(optim, opt_name)(params=learner.parameters(),
                                           **opt_config)
            self.optimizers.append(opt)

    def forward(self, x):
        mus, sigmas = [], []
        for learner in self.learners:  # Sequential prediction
            _, mu, sigma = learner(x)  # mu [ B x 1 x O], sigma [ B x 1 x O]
            mus.append(mu)
            sigmas.append(sigma)
        mus = torch.stack(mus, dim=-1).squeeze(dim=1)  # [ B x O x #.learners ]
        sigmas = torch.stack(sigmas, dim=-1).squeeze(dim=1)  # [ B x O x #.learners ]

        ensemble_mean = mus.mean(dim=-1)  # [ B x O]
        ensemble_var = (sigmas.pow(2) + mus.pow(2)).mean(dim=-1) - ensemble_mean.pow(2)
        ensemble_sigma = ensemble_var.sqrt()
        return ensemble_mean, ensemble_sigma

    def fit(self, train_data, fit_config, val_data=None):
        if fit_config['parallel']:  # Train base learners in parallel
            raise NotImplementedError("Not implemented yet")
        else:  # Train base learners sequentially
            fit_info = self._fit_sequential(train_data=train_data,
                                            fit_config=fit_config,
                                            val_data=val_data)
        return fit_info

    def _fit_sequential(self, train_data, fit_config, val_data=None):
        train_loader = DataLoader(TensorDataset(*train_data),
                                  # shuffle option must be true! Please refer Section 2.4 of deep ensemble paper
                                  shuffle=True,
                                  batch_size=fit_config['batch_size'])

        if val_data is not None:
            val_loader = DataLoader(TensorDataset(*val_data),
                                    shuffle=False,
                                    batch_size=fit_config['val_batch_size'])

        log = dict()
        for learner_idx, (learner, opt) in enumerate(zip(self.learners, self.optimizers)):
            learner.train()
            learner.to(fit_config['device'])

            train_nlls = {'step': [], 'nll': []}
            val_nlls = {'step': [], 'nll': []}

            n_grads = 0
            for i in range(fit_config['epochs']):
                for train_x, train_y in train_loader:
                    train_x = train_x.to(fit_config['device'])
                    train_y = train_y.to(fit_config['device'])

                    if fit_config['adv_eps'] > 0.0:  # adversarial training
                        def adversarial_grad(x):
                            pi, mu, sigma = learner(x)
                            return mdn_loss(pi, mu, sigma, train_y)

                        grad_x = jacobian(adversarial_grad, train_x)
                        adv_x = train_x.clone().detach() + fit_config['adv_eps'] * grad_x
                        adv_y = train_x.clone().detach()

                        train_x = torch.cat([train_x, adv_x], dim=0)
                        train_y = torch.cat([train_y, adv_y], dim=0)

                    pi, mu, sigma = learner(train_x)
                    loss = mdn_loss(pi, mu, sigma, train_y)

                    train_nlls['step'].append(n_grads)
                    train_nlls['nll'].append(loss.item())

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    n_grads += 1

                    if val_data is not None:
                        if n_grads % fit_config['validate_every'] == 0:
                            learner.eval()
                            sum_nll, num_samples = 0.0, 0.0
                            with torch.no_grad():
                                for val_x, val_y in val_loader:
                                    val_x = val_x.to(fit_config['device'])
                                    val_y = val_y.to(fit_config['device'])

                                    val_loss = mdn_loss(*learner(val_x), val_y)
                                    sum_nll += (val_loss * val_x.shape[0]).item()
                                    num_samples += val_x.shape[0]
                            val_nlls['step'].append(n_grads)
                            val_nlls['nll'].append(sum_nll / num_samples)
                            learner.train()
            learner_log = {'train_nlls': train_nlls, 'val_nlls': val_nlls}
            log['Learner{}'.format(learner_idx)] = learner_log
        self.learners.to('cpu')
        return log
