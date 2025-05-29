from Sub_Functions.MultiLayerNet import *
from Sub_Functions.InternalEnergy import *
from Sub_Functions.IntegrationFext import *
from torch.autograd import grad
from hyperopt.early_stop import no_progress_loss

import numpy as np
import pyspark
import time
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy.random as npr
import random
import os
import hyperopt as hopt

npr.seed(2019)
torch.manual_seed(2019)

#------------------------- Constant Network Parameters ----------------
D_in = 2
D_out = 2

# -------------------------- Structural Parameters ---------------------
Length = 4
Height = 1
Depth = 1.0

# -------------------------- Boundary Conditions ------------------------
known_left_ux = 0
known_left_uy = 0
bc_left_penalty = 1.0

known_right_tx = 0
known_right_ty = -10
bc_right_penalty = 1.0

# -------------------------- Material Parameters -----------------------
model_energy = 'Elastic2D'
E = 1000
nu = 0.3

# ------------------------- Datapoints for training ---------------------
Nx = 100
Ny = 25
x_min, y_min = (0.0, 0.0)
hx = Length / (Nx - 1)
hy = Height / (Ny - 1)
shape = [Nx, Ny]
dxdy = [hx, hy]

# ------------------------- Datapoints for evaluation -----------------
Length_test = 4
Height_test = 1
num_test_x = 201
num_test_y = 100
hx_test = Length / (num_test_x - 1)
hy_test = Height / (num_test_y - 1)
shape_test = [num_test_x, num_test_y]

dev = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    device_string = 'cuda'
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    device_string = 'cpu'
    print("CUDA not available, running on CPU")
    torch.set_default_tensor_type('torch.DoubleTensor')

mpl.rcParams['figure.dpi'] = 350

def get_Train_domain():
    x_dom = x_min, Length, Nx
    y_dom = y_min, Height, Ny
    lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
    lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
    dom = np.zeros((Nx * Ny, 2))
    c = 0
    node_dy = (y_dom[1] - y_dom[0]) / (y_dom[2] - 1)
    node_dx = (x_dom[1] - x_dom[0]) / (x_dom[2] - 1)
    for x in np.nditer(lin_x):
        tb = y_dom[2] * c
        te = tb + y_dom[2]
        c += 1
        dom[tb:te, 0] = x
        dom[tb:te, 1] = lin_y

    bcl_u_pts_idx = np.where(dom[:, 0] == x_min)
    bcl_u_pts = dom[bcl_u_pts_idx, :][0]
    bcl_u = np.ones(np.shape(bcl_u_pts)) * [known_left_ux, known_left_uy]

    bcr_t_pts_idx = np.where(dom[:, 0] == Length)
    bcr_t_pts = dom[bcr_t_pts_idx, :][0]
    bcr_t = np.ones(np.shape(bcr_t_pts)) * [known_right_tx, known_right_ty]

    boundary_neumann = {
        "neumann_1": {
            "coord": bcr_t_pts,
            "known_value": bcr_t,
            "penalty": bc_right_penalty,
            "idx": np.asarray(bcr_t_pts_idx)
        }
    }

    boundary_dirichlet = {
        "dirichlet_1": {
            "coord": bcl_u_pts,
            "known_value": bcl_u,
            "penalty": bc_left_penalty,
            "idx": np.asarray(bcl_u_pts_idx)
        }
    }

    bcr_t_pts_idx_new = np.where((dom[:, 0] == Length) & (dom[:, 1] > 0.75))
    bcr_t_pts_new = dom[bcr_t_pts_idx_new, :][0]
    bcr_t_new = np.ones(np.shape(bcr_t_pts_new)) * [known_right_tx, known_right_ty]

    boundary_neumann_new = {
        "neumann_1": {
            "coord": bcr_t_pts_new,
            "known_value": bcr_t_new,
            "penalty": bc_right_penalty,
            "idx": np.asarray(bcr_t_pts_idx_new)
        }
    }

    dom = torch.from_numpy(dom).float()  # Cast to float

    return dom, boundary_neumann, boundary_dirichlet

def get_Test_datatest(Nx=num_test_x, Ny=num_test_y):
    x_dom_test = x_min, Length_test, Nx
    y_dom_test = y_min, Height_test, Ny
    x_space = np.linspace(x_dom_test[0], x_dom_test[1], x_dom_test[2])
    y_space = np.linspace(y_dom_test[0], y_dom_test[1], y_dom_test[2])
    xGrid, yGrid = np.meshgrid(x_space, y_space)
    data_test = np.concatenate(
        (np.array([xGrid.flatten()]).T, np.array([yGrid.flatten()]).T), axis=1)
    return x_space, y_space, data_test

def get_density():
    density = torch.ones(Ny-1, Nx-1)
    train_x_coord = np.transpose(dom[:, 0].reshape(Nx, Ny))
    train_y_coord = np.transpose(dom[:, 1].reshape(Nx, Ny))

    Crcl_x = Length / 2
    Crcl_y = Height / 2
    E_major = 0.25
    E_minor = 0.25

    for nodex in range(Nx):
        for nodey in range(Ny):
            if (((train_x_coord[nodey, nodex] - Crcl_x) / E_major) ** 2 + ((train_y_coord[nodey, nodex] - Crcl_y) / E_minor) ** 2 < 1):
                density[nodey, nodex] = 0

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

class DeepEnergyMethod:
    def __init__(self, model, dim, E, nu, act_func, CNN_dev, rff_dev, N_Layers):
        self.model = MultiLayerNet(model[0], model[1], model[2], act_func, CNN_dev, rff_dev, N_Layers)
        self.model = self.model.to(dev)
        self.InternalEnergy = InternalEnergy(E, nu)
        self.FextLoss = IntegrationFext(dim)
        self.dim = dim
        self.lossArray = []

    def train_model(self, shape, dxdydz, data, neumannBC, dirichletBC, iteration, learning_rate, N_Layers, activatn_fn, density):
        x = data.double().to(dev)  # Ensure data is double
        x.requires_grad_(True)

        dirBC_coordinates = {}
        dirBC_values = {}
        dirBC_penalty = {}
        for i, keyi in enumerate(dirichletBC):
            dirBC_coordinates[i] = torch.from_numpy(dirichletBC[keyi]['coord']).double().to(dev)
            dirBC_values[i] = torch.from_numpy(dirichletBC[keyi]['known_value']).double().to(dev)
            dirBC_penalty[i] = torch.tensor(dirichletBC[keyi]['penalty']).double().to(dev)

        neuBC_coordinates = {}
        neuBC_values = {}
        neuBC_penalty = {}
        neuBC_idx = {}
        for i, keyi in enumerate(neumannBC):
            neuBC_coordinates[i] = torch.from_numpy(neumannBC[keyi]['coord']).double().to(dev)
            neuBC_coordinates[i].requires_grad_(True)
            neuBC_values[i] = torch.from_numpy(neumannBC[keyi]['known_value']).double().to(dev)
            neuBC_penalty[i] = torch.tensor(neumannBC[keyi]['penalty']).double().to(dev)
            neuBC_idx[i] = torch.from_numpy(neumannBC[keyi]['idx']).double().to(dev)

        optimizer_LBFGS = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate, max_iter=20, line_search_fn='strong_wolfe')
        start_time = time.time()
        loss_history = np.zeros(iteration)

        for t in range(iteration):
            def closure():
                u_pred = self.getU(x, N_Layers, activatn_fn)
                storedEnergy = self.InternalEnergy.Elastic2DGauusQuad(u_pred, x, dxdydz, shape, density)
                externalE = self.FextLoss.lossFextEnergy(u_pred, x, neuBC_coordinates, neuBC_values, neuBC_idx, dxdydz)
                energy_loss = storedEnergy - externalE

                bc_u_crit = torch.zeros((len(dirBC_coordinates)))
                for i, vali in enumerate(dirBC_coordinates):
                    dir_u_pred = self.getU(dirBC_coordinates[i], N_Layers, activatn_fn)
                    bc_u_crit[i] = self.loss_squared_sum(dir_u_pred, dirBC_values[i])

                boundary_loss = torch.sum(bc_u_crit)
                loss = energy_loss + boundary_loss
                optimizer_LBFGS.zero_grad()
                loss.backward()
                loss_history[t] = loss.item()
                self.lossArray.append(loss.data.cpu().detach().numpy())
                return loss

            if t > 0 and (np.abs(loss_history[t - 1] - loss_history[t - 2]) < 10e-5):
                break

            optimizer_LBFGS.step(closure)
        elapsed = time.time() - start_time
        return loss_history[t - 1]

    def getU(self, x, N_Layers, activatn_fn):
        u = self.model(x.double(), N_Layers, activatn_fn).double()  # Ensure x is double
        Ux = x[:, 0] * u[:, 0]
        Uy = x[:, 0] * u[:, 1]
        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)
        u_pred = torch.cat((Ux, Uy), -1)
        return u_pred

    @staticmethod
    def loss_squared_sum(tinput, target):
        row, column = tinput.shape
        loss = 0
        for j in range(column):
            loss += torch.sum((tinput[:, j] - target[:, j]) ** 2) / tinput[:, j].data.nelement()
        return loss

def hyperopt_main(x_var):
    lr = x_var['x_lr']
    neuron = int(x_var['neuron'])
    CNN_dev = x_var['CNN_dev']
    rff_dev = x_var['rff_dev']
    iteration = 100
    N_Layers = int(x_var['N_Layers'])
    act_func = x_var['act_func']

    dom, boundary_neumann, boundary_dirichlet = get_Train_domain()
    x, y, datatest = get_Test_datatest()
    density = 1

    dem = DeepEnergyMethod([D_in, neuron, D_out], 2, E, nu, act_func, CNN_dev, rff_dev, N_Layers)
    Loss = dem.train_model(shape, dxdy, dom, boundary_neumann, boundary_dirichlet, iteration, lr, N_Layers, act_func, density)

    print('lr: %.10e\t neuron: %.3d\t CNN_Sdev: %.10e\t RNN_Sdev: %.10e\t Iterations: %.3d\t Layers: %d\t Act_fn: %s\t Loss: %.10e'
          % (lr, neuron, CNN_dev, rff_dev, iteration, N_Layers, act_func, Loss))
    
    # Save the model if it's the best so far
    if not hasattr(hyperopt_main, 'best_loss') or Loss < hyperopt_main.best_loss:
        hyperopt_main.best_loss = Loss
        save_model(dem.model, 'best_model.pth')
        
        # Save the loss history for the best model
        np.save('best_loss_history.npy', dem.lossArray)
        
        # Save the hyperparameters for the best model
        best_params = {
            'lr': lr,
            'neuron': neuron,
            'CNN_dev': CNN_dev,
            'rff_dev': rff_dev,
            'N_Layers': N_Layers,
            'act_func': act_func
        }
        np.save('best_params.npy', best_params)
    
    return Loss

def main(optimize=False, user_params=None):
    if optimize:
        filename = "hyper_opt_results.txt"
        if os.path.exists(filename):
            os.remove(filename)

        space = {
            'x_lr': hopt.hp.loguniform('x_lr', 0, 0.5),
            'neuron': 2 * hopt.hp.quniform('neuron', 10, 60, 1),
            'CNN_dev': hopt.hp.uniform('CNN_dev', 0, 0.2),
            'rff_dev': hopt.hp.uniform('rff_dev', 0, 0.5),
            'N_Layers': hopt.hp.quniform('N_Layers', 1, 10, 1),
            'act_func': hopt.hp.choice('act_func', ['relu', 'tanh'])
        }

        def customStopCondition(x, *kwargs):
            if trials_name.trials[0]["result"]["status"] == "new":
                return False, kwargs
            else:
                return len(trials_name.trials) - 1 - trials_name.best_trial['tid'] > 50, kwargs

        parallel = 0
        if parallel == 1:
            que_len = 5
            trials_name = hopt.SparkTrials(parallelism=que_len)
        else:
            trials_name = hopt.Trials()
            que_len = 1

        # Initialize the best loss
        hyperopt_main.best_loss = float('inf')

        Hopt_strt_time = time.time()
        best = hopt.fmin(hyperopt_main, space, algo=hopt.tpe.suggest, max_evals=200, trials=trials_name, rstate=np.random.default_rng(2019), early_stop_fn=customStopCondition, max_queue_len=que_len)
        Hopt_total_time = time.time() - Hopt_strt_time
        print(best)

        with open(filename, "a") as f:
            f.writelines('Time %.3d\t' % (Hopt_total_time))
            f.writelines('\n')
            f.writelines('lr \t neuron \t CNN_dev \t rff_dev \t N_Layers \t act_func \t Loss \n')
            for opt_iterNo in range(len(trials_name._ids)):
                f.writelines('%.6e\t %.3d\t %.6e\t %.6e\t %.3d\t %s\t %.6e \n' % (
                    trials_name.idxs_vals[1]['x_lr'][opt_iterNo],
                    trials_name.idxs_vals[1]['neuron'][opt_iterNo] * 2,
                    trials_name.idxs_vals[1]['CNN_dev'][opt_iterNo],
                    trials_name.idxs_vals[1]['rff_dev'][opt_iterNo],
                    trials_name.idxs_vals[1]['N_Layers'][opt_iterNo],
                    trials_name.idxs_vals[1]['act_func'][opt_iterNo],
                    trials_name.results[opt_iterNo]['loss']))
            f.writelines('total time= %.2e' % (Hopt_total_time))
        print('total time= %.2e' % (Hopt_total_time))
    else:
        if user_params is None:
            user_params = {
                'x_lr': 0.01,
                'neuron': 20,
                'CNN_dev': 0.1,
                'rff_dev': 0.3,
                'iteration': 100,
                'N_Layers': 5,
                'act_func': 'tanh'
            }
        lr = user_params['x_lr']
        neuron = user_params['neuron']
        CNN_dev = user_params['CNN_dev']
        rff_dev = user_params['rff_dev']
        iteration = user_params['iteration']
        N_Layers = user_params['N_Layers']
        act_func = user_params['act_func']

        dom, boundary_neumann, boundary_dirichlet = get_Train_domain()
        x, y, datatest = get_Test_datatest()
        density = 1

        dem = DeepEnergyMethod([D_in, neuron, D_out], 2, E, nu, act_func, CNN_dev, rff_dev, N_Layers)
        Loss = dem.train_model(shape, dxdy, dom, boundary_neumann, boundary_dirichlet, iteration, lr, N_Layers, act_func, density)

        print('lr: %.10e\t neuron: %.3d\t CNN_Sdev: %.10e\t RNN_Sdev: %.10e\t Iterations: %.3d\t Layers: %d\t Act_fn: %s\t Loss: %.10e'
              % (lr, neuron, CNN_dev, rff_dev, iteration, N_Layers, act_func, Loss))

if __name__ == "__main__":
    # Change the value of 'optimize' to True to perform hyperparameter optimization
    # Change the 'user_params' dictionary to set your own hyperparameters
    optimize = True
    user_params = {
        'x_lr': 0.01,
        'neuron': 20,
        'CNN_dev': 0.1,
        'rff_dev': 0.3,
        'iteration': 1000,
        'N_Layers': 5,
        'act_func': 'tanh'
    }

    main(optimize=optimize, user_params=None)
