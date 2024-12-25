
import torch
import torch.nn as nn
import gpytorch
import math
import matplotlib.pyplot as plt
import linear_operator


def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def plot(model, train_x, train_y):
    # Initialize plots
    f, ((y1_ax, y2_ax), (y3_ax, y4_ax)) = plt.subplots(2, 2, figsize=(8, 8))
    # Test points every 0.02 in [0,1]

    # Make predictions
    with torch.no_grad():
        test_x = torch.linspace(0, 1, 51).view(1, -1, 1).repeat(4, 1, 1)
        test_x = test_x.cuda()
        # print(model.train_inputs[0].shape, test_x[0].shape)
        observed_pred = likelihood(model(test_x[0]))
        # Get mean
        mean = observed_pred.mean
        # Get lower and upper confidence bounds
        lower, upper = observed_pred.confidence_region()

    train_x, train_y = train_x.cpu(), train_y.cpu()
    mean, lower, upper = mean.cpu(), lower.cpu(), upper.cpu()
    test_x = test_x.cpu()

    # Plot training data as black stars
    y1_ax.plot(train_x[0].detach().numpy(), train_y[0].detach().numpy(), 'k*')
    # Predictive mean as blue line
    y1_ax.plot(test_x[0].squeeze().numpy(), mean[0, :].numpy(), 'b')
    # Shade in confidence 
    y1_ax.fill_between(test_x[0].squeeze().numpy(), lower[0, :].numpy(), upper[0, :].numpy(), alpha=0.5)
    y1_ax.set_ylim([-3, 3])
    y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y1_ax.set_title('Observed Values (Likelihood)')

    y2_ax.plot(train_x[1].detach().numpy(), train_y[1].detach().numpy(), 'k*')
    y2_ax.plot(test_x[1].squeeze().numpy(), mean[1, :].numpy(), 'b')
    y2_ax.fill_between(test_x[1].squeeze().numpy(), lower[1, :].numpy(), upper[1, :].numpy(), alpha=0.5)
    y2_ax.set_ylim([-3, 3])
    y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y2_ax.set_title('Observed Values (Likelihood)')

    y3_ax.plot(train_x[2].detach().numpy(), train_y[2].detach().numpy(), 'k*')
    y3_ax.plot(test_x[2].squeeze().numpy(), mean[2, :].numpy(), 'b')
    y3_ax.fill_between(test_x[2].squeeze().numpy(), lower[2, :].numpy(), upper[2, :].numpy(), alpha=0.5)
    y3_ax.set_ylim([-3, 3])
    y3_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y3_ax.set_title('Observed Values (Likelihood)')

    y4_ax.plot(train_x[3].detach().numpy(), train_y[3].detach().numpy(), 'k*')
    y4_ax.plot(test_x[3].squeeze().numpy(), mean[3, :].numpy(), 'b')
    y4_ax.fill_between(test_x[3].squeeze().numpy(), lower[3, :].numpy(), upper[3, :].numpy(), alpha=0.5)
    y4_ax.set_ylim([-3, 3])
    y4_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y4_ax.set_title('Observed Values (Likelihood)')

    plt.show()
    

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act_fn=nn.ELU()):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act_fn = act_fn
        
        self.apply(orthogonal_init)

    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        return self.fc2(x)
    

class DKLOVC(gpytorch.models.ExactGP):
    """
    DKL with OVC cache for constant-time update and inference.
    """
    def __init__(
            self, input_dim, hidden_dim, output_dim, likelihood=None,
            num_inducing_points=256, latent_gp_dim=3,
        ):
        # Generate dummy data to initialize the model
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([output_dim]))
        super(DKLOVC, self).__init__(
            train_inputs=None, 
            train_targets=None, 
            likelihood=likelihood
        )

        self.latent_gp_dim = latent_gp_dim
        self.output_dim = output_dim
        
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, output_dim * latent_gp_dim)
        # self.feature_extractor = lambda x : x.view(1, -1).repeat(4, 1).t()
        
        grid_bound = (-1., 1.) # output of feature extractor is in (-1, 1) (due to the Tanh activation)
        grid_bounds = [grid_bound,] * latent_gp_dim
        self.grid_bounds = grid_bounds
        # self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(grid_bound[0], grid_bound[1])
        self.scale_to_bounds = torch.nn.Tanh()

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([output_dim]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=latent_gp_dim,
                batch_shape=torch.Size([output_dim]),
                # num_mixtures=2,
            ),
            batch_shape=torch.Size([output_dim]),
        )

        self.num_inducing_points = num_inducing_points
        self.pseudo_inputs = None

        self.c = torch.nn.Parameter(
            data=torch.zeros(self.output_dim, self.num_inducing_points),
            requires_grad=False
        )
        self.C = torch.nn.Parameter(
            data=torch.zeros(self.output_dim, self.num_inducing_points, self.num_inducing_points),
            requires_grad=False
        )
        self.m_u = torch.nn.Parameter(
            data=torch.zeros(self.output_dim, self.num_inducing_points),
            requires_grad=False
        )
        self.S_u = torch.nn.Parameter(
            data=torch.zeros(self.output_dim, self.num_inducing_points, self.num_inducing_points),
            requires_grad=False
        )
        self.w = torch.nn.Parameter(
            data=torch.zeros(self.output_dim, self.num_inducing_points),
            requires_grad=False
        )

    @torch.no_grad()
    def clear_cache(self):
        if not self.eval():
            raise RuntimeError("Model must be in eval mode to update cache.")

        self.c.fill_(0.0)
        self.C.fill_(0.0)
        self.m_u.fill_(0.0)
        self.S_u.fill_(0.0)
        self.w.fill_(0.0)

        with linear_operator.settings.cholesky_jitter(1e-3):
            self.update_cache(self.train_inputs[0], self.train_targets)

    def set_train_data(self, inputs, targets, strict=True):
        super(DKLOVC, self).set_train_data(inputs, targets, strict)

        self.latent_train_inputs = self.feature_extractor(self.train_inputs[0]).t()
        self.latent_train_inputs = self.latent_train_inputs.reshape(self.output_dim, self.latent_gp_dim, -1).permute(0, 2, 1)

    @torch.no_grad()
    def update_cache(self, x, y):
        if not self.eval():
            raise RuntimeError("Model must be in eval mode to update cache.")
        
        # print(x.shape, y.shape)
        
        x = self.feature_extractor(x).t()
        x = x.reshape(self.output_dim, self.latent_gp_dim, -1).permute(0, 2, 1)
        y = (y - self.mean_module(x)).unsqueeze(-1)
        noise_inv = self.likelihood.noise ** -1

        if self.pseudo_inputs is not None:
            temp_train_inputs = torch.cat([self.pseudo_inputs, x], dim=-2)

            # select new inducing points
            L, pivots = self.covar_module(temp_train_inputs).pivoted_cholesky(
                rank=self.num_inducing_points,
                error_tol=1e-8,
                return_pivots=True
            )
            rank = L.shape[-1]
            pivots = pivots[:,:rank] # (output_dim, rank)
            pivots = pivots.unsqueeze(-1).expand(self.output_dim, rank, self.latent_gp_dim)
            new_pseudo_inputs = temp_train_inputs.gather(1, pivots) # select the inducing points

            # OVC update
            K_uv = self.covar_module(new_pseudo_inputs, x)
            c_x = K_uv.matmul(y).squeeze(-1).to_dense() * noise_inv
            c_proj = self.covar_module(self.pseudo_inputs, self.pseudo_inputs).solve(self.c.unsqueeze(-1))
            c_add = self.covar_module(new_pseudo_inputs, self.pseudo_inputs).matmul(c_proj).squeeze(-1)
            self.c = torch.nn.Parameter(
                data=c_x + c_add,
                requires_grad=False
            )
            C_x = K_uv.matmul(K_uv.transpose(-1, -2)).to_dense() * noise_inv.unsqueeze(-1)
            C_root = linear_operator.root_decomposition(self.C).root.to_dense()
            C_proj_root = self.covar_module(self.pseudo_inputs, self.pseudo_inputs).solve(C_root)
            C_add_root = self.covar_module(new_pseudo_inputs, self.pseudo_inputs).matmul(C_proj_root)
            C_add = C_add_root.matmul(C_add_root.transpose(-1, -2))
            self.C = torch.nn.Parameter(
                data=C_x + C_add,
                requires_grad=False
            )

            # Update pseudo inputs
            self.pseudo_inputs = new_pseudo_inputs

        else:
            # select the inducing points
            L, pivots = self.covar_module(self.latent_train_inputs).pivoted_cholesky(
                rank=self.num_inducing_points,
                error_tol=1e-8,
                return_pivots=True
            )
            rank = L.shape[-1]
            pivots = pivots[:,:rank] # (output_dim, rank)
            pivots = pivots.unsqueeze(-1).expand(self.output_dim, rank, self.latent_gp_dim)
            self.pseudo_inputs = self.latent_train_inputs.gather(1, pivots)

            # OVC update
            K_uv = self.covar_module(self.pseudo_inputs, x)
            self.c = torch.nn.Parameter(
                data=(K_uv.matmul(y).squeeze(-1).to_dense() * noise_inv),
                requires_grad=False
            )
            self.C = torch.nn.Parameter(
                data=K_uv.matmul(K_uv.transpose(-1, -2)).to_dense() * noise_inv.unsqueeze(-1),
                requires_grad=False
            )

        # Compute m_u (do not compute S_u)
        K_uu = self.covar_module(self.pseudo_inputs)
        M = K_uu + self.C
        v = M.solve(self.c.unsqueeze(-1))
        self.m_u = torch.nn.Parameter(
            data=K_uu.matmul(v).squeeze(-1).detach().to_dense(),
            requires_grad=False
        )

        # Compute cache w
        w = K_uu.solve(self.m_u.unsqueeze(-1))
        self.w = torch.nn.Parameter(
            data=w.squeeze(-1).detach().to_dense(),
            requires_grad=False
        )

    @torch.no_grad()
    def mean_inference(self, x):
        x = self.feature_extractor(x).t()
        x = x.reshape(self.output_dim, self.latent_gp_dim, -1).permute(0, 2, 1)

        mean = self.mean_module(x)
        covar = self.covar_module(x, self.pseudo_inputs)

        return mean + covar.matmul(self.w.unsqueeze(-1)).squeeze(-1)

    def forward(self, x):
        """
        Input shape: (input_dim, batch_shape)
        Output mean shape: (output_dim, batch_shape)
        """
        x = self.feature_extractor(x).t()
        x = x.reshape(self.output_dim, self.latent_gp_dim, -1).permute(0, 2, 1)
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    

if __name__ == '__main__':
    # Training data is 100 points in [0,1] inclusive regularly spaced
    train_x = torch.linspace(0, 1, 3000).view(1, -1, 1).repeat(4, 1, 1)
    # True functions are sin(2pi x), cos(2pi x), sin(pi x), cos(pi x)
    sin_y = torch.sin((train_x[0]-0.3) ** 2 * 30) + 0.5 * torch.rand(1, 3000, 1)
    sin_y_short = torch.sinh(train_x[0] * (math.pi)) / 2 + 0.5 * torch.rand(1, 3000, 1)
    cos_y = torch.sin((1-train_x[0])**3 * (2 * math.pi) * 2) + 0.5 * torch.rand(1, 3000, 1)
    cos_y_short = torch.cos(train_x[0]*10).unsqueeze(0) + 0.5 * torch.rand(1, 3000, 1)
    train_y = torch.cat((sin_y, sin_y_short, cos_y, cos_y_short)).squeeze(-1)

    ori_train_x, ori_train_y = train_x.cuda(), train_y.cuda()

    # Split the training data to two parts, one for training, another one for updating
    train_x, update_x = ori_train_x[:, :1000, :], ori_train_x[:, 1000:, :]
    train_y, update_y = ori_train_y[:, :1000], ori_train_y[:, 1000:]

    model = DKLOVC(input_dim=1, hidden_dim=16, output_dim=4).cuda()
    model.set_train_data(inputs=train_x[0], targets=train_y, strict=False)

    model.train()
    likelihood = model.likelihood
    optimizer = torch.optim.Adam(
        [
            # {'params': model.feature_extractor.parameters(), 'lr': 1e-2, 'weight_decay': 1e-3},
            {'params': model.mean_module.parameters()},
            {'params': model.covar_module.parameters()},
            {'params': model.likelihood.parameters()},
        ],
        lr=1e-2
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    training_iter = 500
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x[0])
        loss = -mll(output, train_y).mean()
        loss.backward()
        optimizer.step()

    # Set into eval mode
    model.eval()
    likelihood.eval()

    plot(model, train_x, train_y)

    model.clear_cache()

    print("(Before) Rank: ", model.pseudo_inputs.shape[-2])

    with linear_operator.settings.cholesky_jitter(1e-1):
        # model.update_cache(update_x[0], update_y)
        for i in range(update_x.shape[1] // 500):
            print("Update: ", i)
            model.update_cache(update_x[0,i * 500:(i+1) * 500,:], update_y[:,i * 500:(i+1) * 500])

    print("(After) Rank: ", model.pseudo_inputs.shape[-2])

    # plot the update points (only plot the mean function through mean_inference)
    with torch.no_grad():
        test_x = torch.linspace(0, 1, 51).view(1, -1, 1).repeat(4, 1, 1)
        test_x = test_x.cuda()
        mean = model.mean_inference(test_x[0])

    test_x = test_x.cpu()
    mean = mean.cpu()
    update_x = update_x.cpu()
    update_y = update_y.cpu()

    f, ((y1_ax, y2_ax), (y3_ax, y4_ax)) = plt.subplots(2, 2, figsize=(8, 8))
    y1_ax.plot(train_x[0].detach().cpu().numpy(), train_y[0].detach().cpu().numpy(), 'k*')
    y1_ax.plot(test_x[0].squeeze().numpy(), model.mean_module(test_x)[0].detach().cpu().numpy(), 'g--')
    y1_ax.plot(update_x[0].detach().numpy(), update_y[0].detach().numpy(), 'r*')
    y1_ax.plot(test_x[0].squeeze().numpy(), mean[0, :].numpy(), 'b')
    y1_ax.set_ylim([-3, 3])
    y1_ax.legend(['Observed Data', 'Mean', 'Updated Data', 'Mean Prediction'])
    y1_ax.set_title('Observed Values (Mean)')

    y2_ax.plot(train_x[1].detach().cpu().numpy(), train_y[1].detach().cpu().numpy(), 'k*')
    y2_ax.plot(test_x[1].squeeze().numpy(), model.mean_module(test_x)[1].detach().cpu().numpy(), 'g--')
    y2_ax.plot(update_x[1].detach().numpy(), update_y[1].detach().numpy(), 'r*')
    y2_ax.plot(test_x[1].squeeze().numpy(), mean[1, :].numpy(), 'b')
    y2_ax.set_ylim([-3, 3])
    y2_ax.legend(['Observed Data', 'Mean', 'Updated Data', 'Mean Prediction'])
    y2_ax.set_title('Observed Values (Mean)')

    y3_ax.plot(train_x[2].detach().cpu().numpy(), train_y[2].detach().cpu().numpy(), 'k*')
    y3_ax.plot(test_x[2].squeeze().numpy(), model.mean_module(test_x)[2].detach().cpu().numpy(), 'g--')
    y3_ax.plot(update_x[2].detach().numpy(), update_y[2].detach().numpy(), 'r*')
    y3_ax.plot(test_x[2].squeeze().numpy(), mean[2, :].numpy(), 'b')
    y3_ax.set_ylim([-3, 3])
    y3_ax.legend(['Observed Data', 'Mean', 'Updated Data', 'Mean Prediction'])
    y3_ax.set_title('Observed Values (Mean)')

    y4_ax.plot(train_x[3].detach().cpu().numpy(), train_y[3].detach().cpu().numpy(), 'k*')
    y4_ax.plot(test_x[3].squeeze().numpy(), model.mean_module(test_x)[3].detach().cpu().numpy(), 'g--')
    y4_ax.plot(update_x[3].detach().numpy(), update_y[3].detach().numpy(), 'r*')
    y4_ax.plot(test_x[3].squeeze().numpy(), mean[3, :].numpy(), 'b')
    y4_ax.set_ylim([-3, 3])
    y4_ax.legend(['Observed Data', 'Mean', 'Updated Data', 'Mean Prediction'])
    y4_ax.set_title('Observed Values (Mean)')

    # Plot pivots as red stars
    if hasattr(model, 'pivots'):
        for i, ax in enumerate([y1_ax, y2_ax, y3_ax, y4_ax]):
            pivots = model.pivots[i].cpu().numpy()
            pivot_x = train_x[i, pivots, 0].detach().cpu().numpy()
            for pivot in pivot_x:
                ax.axvline(x=pivot, color='r', linestyle='--')

    plt.show()
