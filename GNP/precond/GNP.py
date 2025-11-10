import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
from torch.utils.data.dataloader import DataLoader
import os
import numpy as np
from tqdm import tqdm

from GNP.solver import Arnoldi
from GNP.utils import load_npzsparse, scale_A_by_spectral_radius

class StreamingDataset_AAA(IterableDataset):
    # A is torch tensor, either sparse or full
    def __init__(self, A, batch_size, m):
        super().__init__()
        self.n = A.shape[0]
        self.m = m
        self.batch_size = batch_size

        arnoldi = Arnoldi()
        Vm1, barHm = arnoldi.build(A, m=m)
        W, S, Zh = torch.linalg.svd(barHm, full_matrices=False)
        Q = ( Vm1[:,:-1] @ Zh.T ) / S.view(1, m)
        self.Q = Q.to('cpu')

    def generate(self):
        while True:
            batch_size1 = self.batch_size // 2
            e = torch.normal(0, 1, size=(self.m, batch_size1),
                                dtype=torch.float64)
            x = self.Q @ e
            batch_size2 = self.batch_size - batch_size1
            x2 = torch.normal(0, 1, size=(self.n, batch_size2),
                                dtype=torch.float64)
            x = torch.cat([x, x2], dim=1)
            yield x
            
    def __iter__(self):
        return iter(self.generate())

class NPZDataset_AAA(Dataset):
    def __init__(self, path_to_matrix, has_solution, m, batch_size):
        super().__init__()
        self.datalist = list(map(int, map(lambda x: x.strip(".npz").strip("b_").strip("c_").strip("f_").strip("r_").strip("s_"), os.listdir(path_to_matrix))))
        self.has_solution = has_solution
        self.path_to_matrix = path_to_matrix
        self.m = m
        self.batch_size = batch_size
        self.AAA = []
        self.bbb = []
        self.xxx = []
        for idx in self.datalist:
            A, b, x = load_npzsparse(self.path_to_matrix, idx, "cpu", self.has_solution)
            self.AAA.append(A)
            self.bbb.append(b)
            self.xxx.append(x)

    def __len__(self):
        return len(self.datalist)
    
    def get_arnoldi_decomp(self, A):
        arnoldi = Arnoldi()
        Vm1, barHm = arnoldi.build(A, m=self.m)
        W, S, Zh = torch.linalg.svd(barHm, full_matrices=False)
        Q = ( Vm1[:,:-1] @ Zh.T ) / S.view(1, self.m)
        Q = Q.to('cpu')
        return Q

    def generate(self, Q, count=10):
        self.n = Q.shape[0]
        data = []
        for i in range(count):
            batch_size1 = self.batch_size // 2
            e = torch.normal(0, 1, size=(self.m, batch_size1),
                                dtype=torch.float64)
            x = Q @ e
            batch_size2 = self.batch_size - batch_size1
            x2 = torch.normal(0, 1, size=(self.n, batch_size2),
                                dtype=torch.float64)
            x = torch.cat([x, x2], dim=1)
            data.append(x)
        return data

    def __getitem__(self, idx):
        #A, b, x = load_npzsparse(self.path_to_matrix, self.datalist[idx], "cpu", self.has_solution)
        A, b, x = self.AAA[idx], self.bbb[idx], self.xxx[idx]
        # Normalize A to avoid hassles
        A, gamma = scale_A_by_spectral_radius(A)
        #print(f"{A.shape=}")
        b = b / gamma
        Q = self.get_arnoldi_decomp(A)
        #print(f"{Q.shape=}")
        data = self.generate(Q, count=10)
        return data, A, b, x

#-----------------------------------------------------------------------------
# Graph neural preconditioner with npz dataset
class GNP_AAA():
    # A is torch tensor, either sparse or full
    def __init__(self, path_to_matrix, has_solution, net, device):
        self.net = net
        self.device = device
        self.dtype = net.dtype
        self.path_to_matrix = path_to_matrix
        self.has_solution = has_solution
        
    def train(self, m, batch_size, grad_accu_steps, epochs, optimizer,
              scheduler=None, num_workers=0, checkpoint_prefix_with_path=None,
              progress_bar=True):

        self.net.train()
        optimizer.zero_grad()

        self.dataset = NPZDataset_AAA(self.path_to_matrix, self.has_solution, m, batch_size)
        #loader = DataLoader(self.dataset, batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
        
        hist_loss = []
        best_loss = np.inf
        best_epoch = -1
        checkpoint_file = None
        
        if progress_bar:
            pbar = tqdm(total=epochs, desc='Train')

        for epoch in range(epochs):
            epoch_loss_sum = 0
            epoch_loss_min = np.inf
            epoch_loss_max = 0
            for i in torch.randperm(len(self.dataset)):
                data, A, b, x = self.dataset[i]
                for x_random in data:
                    #print(f"{i=}, {x_random.shape=}")
                    A = A.to(self.device).to(self.dtype)
                    b_random = x_random.to(self.device).to(self.dtype)
                    # Train
                    x_out = self.net(A, b_random)
                    b_out = (A @ x_out)#.to(torch.float64)).to(self.dtype)
                    loss = F.l1_loss(b_out.squeeze(), b_random.squeeze())

                    # Train (cont.)
                    loss.backward()
                    if (epoch+1) % grad_accu_steps == 0 or epoch == epochs - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()
                    epoch_loss_sum += loss.item()
                    if epoch_loss_min > loss.item():
                        epoch_loss_min = loss.item()
                    if epoch_loss_max < loss.item():
                        epoch_loss_max = loss.item()

                
            # Bookkeeping
            hist_loss.append(epoch_loss_sum)
            if epoch_loss_sum < best_loss:
                best_loss = epoch_loss_sum
                best_epoch = epoch
                if checkpoint_prefix_with_path is not None:
                    checkpoint_file = checkpoint_prefix_with_path + 'best.pt'
                    torch.save(self.net.state_dict(), checkpoint_file)

            # Bookkeeping (cont.)
            if progress_bar:
                pbar.set_description(f'Train epoch_loss_sum {epoch_loss_sum:.1e} min {epoch_loss_min:.1e} max {epoch_loss_max:.1e}')
                pbar.update()
            if epoch == epochs - 1:
                break

        # Bookkeeping (cont.)
        if checkpoint_file is not None:
            checkpoint_file_old = checkpoint_file
            checkpoint_file = \
                checkpoint_prefix_with_path + f'epoch_{best_epoch}.pt'
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            os.rename(checkpoint_file_old, checkpoint_file)
            
        return hist_loss, best_loss, best_epoch, checkpoint_file

    @torch.no_grad()
    def apply(self, A, r): # r: float64
        self.net.eval()
        A = A.to(self.dtype) # -> lower precision
        r = r.to(self.dtype) # -> lower precision
        r = r.view(-1, 1)
        z = self.net(A, r)
        z = z.view(-1)
        z = z.double() # -> float64
        return z


#-----------------------------------------------------------------------------
# Graph neural preconditioner with npz dataset
class GNPNPZ():

    # A is torch tensor, either sparse or full
    def __init__(self, path_to_matrix, has_solution, net, device):
        self.net = net
        self.device = device
        self.dtype = net.dtype
        self.path_to_matrix = path_to_matrix
        self.has_solution = has_solution
        
    def train(self, batch_size, grad_accu_steps, epochs, optimizer,
              scheduler=None, num_workers=0, checkpoint_prefix_with_path=None,
              progress_bar=True):

        self.net.train()
        optimizer.zero_grad()

        self.dataset = NPZDataset(self.path_to_matrix, self.has_solution)
        #loader = DataLoader(self.dataset, batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
        
        hist_loss = []
        best_loss = np.inf
        best_epoch = -1
        checkpoint_file = None
        
        if progress_bar:
            pbar = tqdm(total=epochs, desc='Train')

        for epoch in range(epochs):
            epoch_loss_sum = 0
            epoch_loss_min = np.inf
            epoch_loss_max = 0
            for i in torch.randperm(len(self.dataset)):
                A, b, x = self.dataset[i]
                A = A.to(self.dtype).to(self.device)
                b = b.to(self.dtype).to(self.device)
                # Train
                x_out = self.net(A, b)
                #b_out = (A @ x_out.to(torch.float64)).to(self.dtype)
                b_out = (A @ x_out)
                loss = F.l1_loss(b_out.squeeze(), b)

                # Train (cont.)
                loss.backward()
                if (epoch+1) % grad_accu_steps == 0 or epoch == epochs - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                epoch_loss_sum += loss.item()
                if epoch_loss_min > loss.item():
                    epoch_loss_min = loss.item()
                if epoch_loss_max < loss.item():
                    epoch_loss_max = loss.item()
                
            # Bookkeeping
            hist_loss.append(epoch_loss_sum)
            if epoch_loss_sum < best_loss:
                best_loss = epoch_loss_sum
                best_epoch = epoch
                if checkpoint_prefix_with_path is not None:
                    checkpoint_file = checkpoint_prefix_with_path + 'best.pt'
                    torch.save(self.net.state_dict(), checkpoint_file)

            # Bookkeeping (cont.)
            if progress_bar:
                pbar.set_description(f'Train epoch_loss_sum {epoch_loss_sum:.1e} min {epoch_loss_min:.1e} max {epoch_loss_max:.1e}')
                pbar.update()
            if epoch == epochs - 1:
                break

        # Bookkeeping (cont.)
        if checkpoint_file is not None:
            checkpoint_file_old = checkpoint_file
            checkpoint_file = \
                checkpoint_prefix_with_path + f'epoch_{best_epoch}.pt'
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            os.rename(checkpoint_file_old, checkpoint_file)
            
        return hist_loss, best_loss, best_epoch, checkpoint_file

    @torch.no_grad()
    def apply(self, A, r): # r: float64
        self.net.eval()
        A = A.to(self.dtype) # -> lower precision
        r = r.to(self.dtype) # -> lower precision
        r = r.view(-1, 1)
        z = self.net(A, r)
        z = z.view(-1)
        z = z.double() # -> float64
        return z



#-----------------------------------------------------------------------------
# The following class implements a streaming dataset, which, in
# combined use with the dataloader, produces x of size (n,
# batch_size). x is float64 and stays in cpu. It will be moved to the
# device and cast to a lower precision for training.
class StreamingDataset(IterableDataset):
    # A is torch tensor, either sparse or full
    def __init__(self, A, batch_size, training_data, m):
        super().__init__()
        self.n = A.shape[0]
        self.m = m
        self.batch_size = batch_size
        self.training_data = training_data

        # Computations done in device
        if training_data == 'x_subspace' or training_data == 'x_mix':
            arnoldi = Arnoldi()
            Vm1, barHm = arnoldi.build(A, m=m)
            W, S, Zh = torch.linalg.svd(barHm, full_matrices=False)
            Q = ( Vm1[:,:-1] @ Zh.T ) / S.view(1, m)
            self.Q = Q.to('cpu')

    def generate(self):
        while True:

            # Computation done in cpu
            if self.training_data == 'x_normal':
                
                x = torch.normal(0, 1, size=(self.n, self.batch_size),
                                 dtype=torch.float64)
                yield x

            elif self.training_data == 'x_subspace':

                e = torch.normal(0, 1, size=(self.m, self.batch_size),
                                 dtype=torch.float64)
                x = self.Q @ e
                yield x

            elif self.training_data == 'x_mix':
            
                batch_size1 = self.batch_size // 2
                e = torch.normal(0, 1, size=(self.m, batch_size1),
                                 dtype=torch.float64)
                x = self.Q @ e
                batch_size2 = self.batch_size - batch_size1
                x2 = torch.normal(0, 1, size=(self.n, batch_size2),
                                  dtype=torch.float64)
                x = torch.cat([x, x2], dim=1)
                yield x

            else: # self.training_data == 'no_x'

                b = torch.normal(0, 1, size=(self.n, self.batch_size),
                                 dtype=torch.float64)
                yield b
            
    def __iter__(self):
        return iter(self.generate())


#-----------------------------------------------------------------------------
# Graph neural preconditioner
class GNP():

    # A is torch tensor, either sparse or full
    def __init__(self, A, training_data, m, net, device, ):
        self.A = A
        self.training_data = training_data
        self.m = m
        self.net = net
        self.device = device
        self.dtype = net.dtype

    def train(self, batch_size, grad_accu_steps, epochs, optimizer,
              scheduler=None, num_workers=0, checkpoint_prefix_with_path=None,
              progress_bar=True):

        self.net.train()
        optimizer.zero_grad()
        dataset = StreamingDataset(self.A, batch_size,
                                       self.training_data, self.m)
        loader = DataLoader(dataset, num_workers=num_workers, pin_memory=True)
        
        hist_loss = []
        best_loss = np.inf
        best_epoch = -1
        checkpoint_file = None
            
        if progress_bar:
            pbar = tqdm(total=epochs, desc='Train')

        for epoch, x_or_b in enumerate(loader):

            # Generate training data
            if self.training_data != 'no_x':
                x = x_or_b[0].to(self.device)
                b = self.A @ x
                b, x = b.to(self.dtype), x.to(self.dtype)
            else: # self.training_data == 'no_x'
                b = x_or_b[0].to(self.device).to(self.dtype)

            # Train
            x_out = self.net(b)
            b_out = (self.A @ x_out.to(torch.float64)).to(self.dtype)
            loss = F.l1_loss(b_out, b)

            # Bookkeeping
            hist_loss.append(loss.item())
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_epoch = epoch
                if checkpoint_prefix_with_path is not None:
                    checkpoint_file = checkpoint_prefix_with_path + 'best.pt'
                    torch.save(self.net.state_dict(), checkpoint_file)

            # Train (cont.)
            loss.backward()
            if (epoch+1) % grad_accu_steps == 0 or epoch == epochs - 1:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            # Bookkeeping (cont.)
            if progress_bar:
                pbar.set_description(f'Train loss {loss:.1e}')
                pbar.update()
            if epoch == epochs - 1:
                break

        # Bookkeeping (cont.)
        if checkpoint_file is not None:
            checkpoint_file_old = checkpoint_file
            checkpoint_file = \
                checkpoint_prefix_with_path + f'epoch_{best_epoch}.pt'
            os.rename(checkpoint_file_old, checkpoint_file)
            
        return hist_loss, best_loss, best_epoch, checkpoint_file

    @torch.no_grad()
    def apply(self, r): # r: float64
        self.net.eval()
        r = r.to(self.dtype) # -> lower precision
        r = r.view(-1, 1)
        z = self.net(r)
        z = z.view(-1)
        z = z.double() # -> float64
        return z
