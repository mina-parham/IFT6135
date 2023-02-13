import torch
from q2_sampler import svhn_sampler
from q2_model import Critic, Generator
from torch import optim
from torchvision.utils import save_image



def lp_reg(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. Also consider that the norm used is the L2 norm. This is important to consider,
    because we make the assumption that your implementation follows this notation when testing your function. ***

    :param x: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution P.
    :param y: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution Q.
    :param critic: (Module) - torch module that you want to regularize.
    :return: (FloatTensor) - shape: (1,) - Lipschitz penalty
    """
    # batch_size = x.size(0)
    # dim = x.size(1)
    # t = torch.rand(batch_size,1)
    # t = t.expand(batch_size, int(x.nelement()/batch_size)).contiguous().view(batch_size, 3, 32, 32)
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    print(shape)
    t = torch.rand(shape)
    x_hat = t*x + (1-t)*y
    x_hat = torch.autograd.Variable(x_hat, requires_grad=True)
    f = critic(x_hat)

    grad_f = torch.autograd.grad(outputs=f, inputs=x_hat,
                               grad_outputs=torch.ones(f.size()),
                               create_graph=True, retain_graph=True)[0]
    

    gradients = grad_f.view(x_hat.size(0), -1)

    test = (gradients.norm(2, dim=1)-1)
    for i in range(len(test)):
      if test[i] < 0:
        test[i] = 0
    
    return (test **2).mean()


def vf_wasserstein_distance(p, q, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. This is important to consider, because we make the assuption that your implementation
    follows this notation when testing your function. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Wasserstein distance
    :return: (FloatTensor) - shape: (1,) - Estimate of the Wasserstein distance
    """
    out = critic(p).mean() - critic(q).mean()
    return out



if __name__ == '__main__':
    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    data_root = './'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_iter = 50000 # N training iterations
    n_critic_updates = 5 # N critic updates per generator update
    lp_coeff = 10 # Lipschitz penalty coefficient
    train_batch_size = 64
    test_batch_size = 64
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    z_dim = 100

    train_loader, valid_loader, test_loader = svhn_sampler(data_root, train_batch_size, test_batch_size)

    generator = Generator(z_dim=z_dim).to(device)
    critic = Critic().to(device)

    optim_critic = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

    # COMPLETE TRAINING PROCEDURE
    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)
    test_iter = iter(test_loader)
    for i in range(n_iter):
        generator.train()
        critic.train()
        for _ in range(n_critic_updates):
            try:
                data = next(train_iter)[0].to(device)
            except Exception:
                train_iter = iter(train_loader)
                data = next(train_iter)[0].to(device)
            #####
            # train the critic model here
            optim_critic.zero_grad()
            data_fake = generator(data)
            reg = lp_reg(data, data_fake, critic)
            loss = vf_wasserstein_distance(data_fake, data, critic)
            main_loss = loss + reg * lp_coeff
            if(i % 2000 == 0):
              print("Iter is: ", i, "  Loss discriminator: ", main_loss.item())

            main_loss.backward()
            optim_critic.step()



            #####

        #####
        # train the generator model here
        optim_generator.zero_grad()
        data_fake = generator(data)
        g_loss =- torch.mean(critic(data_fake))
        if(i % 2000 == 0):
              print("Iter is: ", i, "  Loss generator: ", g_loss.item())

        loss.backward()
        optim_generator.step()

        #####

        # Save sample images 
        if i % 100 == 0:
            z = torch.randn(64, z_dim, device=device)
            imgs = generator(z)
            save_image(imgs, f'imgs_{i}.png', normalize=True, value_range=(-1, 1))


    # COMPLETE QUALITATIVE EVALUATION

