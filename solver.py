import sys
import time
import torch
from scipy.sparse import csr_matrix
import numpy as np
import warnings
from time import perf_counter
from tqdm import tqdm
import warnings
from warnings import warn
# Suppress the specific UserWarning
warnings.filterwarnings('ignore', message='.*Sparse CSC tensor support is in beta state.*')
warnings.filterwarnings('ignore', message='.*Sparse CSR tensor support is in beta state.*')

#-----------------------------------------------------------------------------
# Scale A by rmse norm of eah raw
def normalize_rows(A):
    rows_norms = np.sqrt(np.add.reduceat(A.data**2, A.indptr[:-1])) / np.diff(A.indptr)
    A.data = A.data / np.repeat(rows_norms, np.diff(A.indptr))
    return A, rows_norms

#-----------------------------------------------------------------------------
# Scale A by an estimated spectral radius according to the Gershgorin circle theorem.
def scale_A_by_spectral_radius(A):
    if A.layout == torch.sparse_csc or A.layout == torch.sparse_csr:
        absA = torch.absolute(A)
        m, n = absA.shape
        row_sum = absA @ torch.ones(n, 1, dtype=A.dtype, device=A.device)
        col_sum = torch.ones(1, m, dtype=A.dtype, device=A.device) @ absA
        gamma = torch.min(torch.max(row_sum), torch.max(col_sum)).item()
        outA = A * (1. / gamma)
        
    elif A.layout == torch.strided:
        absA = torch.absolute(A)
        row_sum = torch.sum(absA, dim=1)
        col_sum = torch.sum(absA, dim=0)
        gamma = torch.min(torch.max(row_sum), torch.max(col_sum))
        outA = A / gamma
        
    else:
        raise NotImplementedError(
            'A must be either torch.sparse_csc_tensor or torch.tensor')
    
    return outA, gamma

# Convention: M \approx inv(A)
#-----------------------------------------------------------------------------
# (Right-)Preconditioned generalized minimal residual method. Can use
# flexible preconditioner. Current implementation supports only a
# single right-hand side.
# If timeout is not None, max_iters is disabled.
class GMRES():
    def solve(self, A, b, M=None, x0=None, restart=10, max_iters=100,
              timeout=None, rtol=1e-8, progress_bar=True):
        if progress_bar:
            if timeout is None:
                pbar = tqdm(total=max_iters, desc='Solve')
                pbar.update()
            else:
                pbar = tqdm(desc='Solve')
                pbar.update()
                        
        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0
        norm_b = torch.linalg.norm(b)
        hist_abs_res = []
        hist_rel_res = []
        hist_time = []

        n = len(b)
        V = torch.zeros(n, restart+1, dtype=b.dtype).to(b.device)
        Z = torch.zeros(n, restart, dtype=b.dtype).to(b.device)
        H = torch.zeros(restart+1, restart, dtype=b.dtype).to(b.device)
        g = torch.zeros(restart+1, dtype=b.dtype).to(b.device)
        c = torch.zeros(restart, dtype=b.dtype).to(b.device)
        s = torch.zeros(restart, dtype=b.dtype).to(b.device)
        
        tic = time.time()

        # Initial step
        r = b - A @ x
        beta = torch.linalg.norm(r)
        abs_res = beta
        rel_res = abs_res / norm_b
        hist_abs_res.append(abs_res.item())
        hist_rel_res.append(rel_res.item())
        hist_time.append(time.time() - tic)
        iters = 0
        quit_iter = False

        # Outer loop
        while 1:
            # Restart cycle
            V[:,0] = r / beta
            g[0] = beta
            for j in range(restart):
                if M is not None:
                    Z[:,j] = M.apply(A, V[:,j]) # единственная изменённая строчка в GMRES_AAA по сравнению с GMRES
                else:
                    Z[:,j] = V[:,j]
                w = A @ Z[:,j]
                for k in range(j+1):
                    H[k,j] = torch.dot(V[:,k], w)
                    w = w - H[k,j] * V[:,k]
                H[j+1,j] = torch.linalg.norm(w)
                V[:,j+1] = w / H[j+1,j]

                # Solve min || H * y - beta * e1 ||: Givens rotation
                for k in range(j):
                    tmp      =  c[k] * H[k,j] + s[k] * H[k+1,j]
                    H[k+1,j] = -s[k] * H[k,j] + c[k] * H[k+1,j]
                    H[k,j] = tmp
                t = torch.sqrt( H[j,j]**2 + H[j+1,j]**2 )
                c[j], s[j] = H[j,j]/t, H[j+1,j]/t
                H[j,j] = c[j] * H[j,j] + s[j] * H[j+1,j]
                H[j+1,j] = 0
                g[j+1] = -s[j] * g[j]
                g[j] = c[j] * g[j]
                # End solve min || H * y - beta * e1 ||: Givens rotation

                abs_res = torch.abs(g[j+1])
                rel_res = abs_res / norm_b
                hist_abs_res.append(abs_res.item())
                hist_rel_res.append(rel_res.item())
                hist_time.append(time.time() - tic)
                iters = iters + 1
                if (rel_res < rtol) or \
                   (timeout is None and iters == max_iters) or \
                   (timeout is not None and hist_time[-1] >= timeout):
                    quit_iter = True
                    break

                if progress_bar:
                    pbar.update()
            # End restart cycle

            # Solve min || H * y - beta * e1 ||: obtain solution
            y = torch.linalg.solve_triangular(H[:j+1, :j+1],
                                              g[:j+1].view(j+1,1),
                                              upper=True).view(j+1)
            # End solve min || H * y - beta * e1 ||: obtain solution

            x = x + Z[:, :j+1] @ y
            r = b - A @ x
            beta = torch.linalg.norm(r)
            if np.allclose(hist_abs_res[-1], beta.item()) is False:
                warn('Residual tracked by least squares solve is different '
                     'from the true residual. The result of GMRES should not '
                     'be trusted.')
            if quit_iter == True:
                break
            else:
                g = g.zero_()
                g[0] = beta
        # End outer loop

        if progress_bar:
            pbar.close()

        return x, iters, hist_abs_res, hist_rel_res, hist_time

# Graph neural preconditioner with npz dataset
class GNP():
    def __init__(self, net, dtype=torch.float32):
        self.net = net
        self.dtype = dtype
    @torch.no_grad()
    def apply(self, A, r): # r: float64
        self.net.eval()
        A = A.type(self.dtype) # -> lower precision
        r = r.type(self.dtype) # -> lower precision
        r = r.view(-1, 1)
        z = self.net(A, r)
        z = z.view(-1)
        z = z.double() # -> float64
        return z

def solve_gmres_GNP(A, b, rtol=1e-8, maxiter=1000, restart=30):
    """
    Решает систему Ax = b с помощью GMRES с предобуславливателем на основе
    нейронной сети GNP (Graph Neural Preconditioner)

    Параметры:
    ----------
    A : sparse matrix
        Матрица системы.
    b : array
        Правая часть системы.
    rtol : float
        Относительная точность для GMRES.
    maxiter : int
        Максимальное число итераций GMRES.
    restart : int
        Число итераций до перезапуска GMRES.
    aggregation_params : dict or None
        Параметры для pyamg.smoothed_aggregation_solver.
        Если None, используются значения по умолчанию.

    Возвращает:
    -----------
    x : array
        Приближённое решение.
    iteration_count : int
        Число итераций GMRES.
    setup_time : float
        Время настройки предобуславливателя.
    solve_time : float
        Время решения GMRES.
    """
    # setup start
    setup_start = perf_counter()

    n = A.shape[0]
    timeout = None              # timeout in seconds
    hide_solver_bar = True     # whether hide progress bar in linear solver
    GNP_dtype = torch.float32       # точность для вычислений нейросетью
    
    model_path = "checkpoint"
    choice_of_model = { '0': (1, 1450),
                        '1': (1451, 3394),
                        '2': (3395, 5284),
                        '3': (5285, 7826),
                        '4': (7927, 10077),
                        '5': (10078, 10555),
                        '6': (10556, 10830),
                        '7': (10831, 11000)}
    for p in choice_of_model:
        if choice_of_model[p][0] <= n <= choice_of_model[p][1]:
            model_path = f"./checkpoints/scriptmodule_{p}.ts"
            break
    # Computing device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    # # GMRES without preconditioner
    # print('\nSolving linear system without preconditioner ...')
    # _, _, _, hist_rel_res, hist_time = solver.solve(
    #     A, b, M=None, restart=restart, max_iters=max_iters,
    #     timeout=timeout, rtol=rtol, progress_bar=not hide_solver_bar)
    # print(f'Done. Final relative residual = {hist_rel_res[-1]:.4e}')

    # Настройка предобуславливателя на основе нейросети
    net = torch.jit.load(model_path, map_location=device)
    M = GNP(net, GNP_dtype)   
    solver = GMRES()

    setup_time = perf_counter() - setup_start

    # iteration_count = [0]
    # def callback(_):
    #     iteration_count[0] += 1

    # solve start
    solve_start = perf_counter()
    
    # Normalize A to avoid hassles
    A, rows_norms = normalize_rows(A)
    b = b / rows_norms
    A = torch.sparse_csr_tensor(A.indptr, A.indices, A.data, A.shape,
                                dtype=torch.float64).to(device)
    A = A.to_sparse_csc()
    b = torch.tensor(b, dtype=torch.float64).to(device)

    A, gamma = scale_A_by_spectral_radius(A)
    b = b / gamma
    A = A.to(device)
    b = b.to(device)
    try:
        x, iters, hist_abs_res, hist_rel_res_gnp, hist_time_gnp = solver.solve(
        A, b, M=M, restart=restart, max_iters=maxiter,
        timeout=timeout, rtol=rtol, progress_bar=not hide_solver_bar)
    except UserWarning as w:
        print('Warning:', w)
        print('GMRES preconditioned by GNP fails')
        hist_rel_res_gnp = None
        hist_time_gnp = None
    else:
        print(f'Done. Final relative residual = {hist_rel_res_gnp[-1]:.4e}')
    warnings.resetwarnings()
    # x, info = gmres(
    #     A,
    #     b,
    #     rtol=rtol,
    #     maxiter=maxiter,
    #     restart=restart,
    #     M=M,
    #     callback=callback,
    #     callback_type='legacy',
    #     atol=0
    # )
    solve_time = perf_counter() - solve_start

    # if info > 0:
    #     warnings.warn(f"GMRES did not converge (info={info}) after {iteration_count[0]} iterations.")
    # elif info < 0:
    #     raise ValueError(f"Illegal input in GMRES (info={info})")

    return x.cpu().detach().numpy(), iters, setup_time, solve_time


def dimension_reduction_solve(F, C, B, r_vec):
    # Решаем систему (C - B F) x = r_vec напрямую используя блоки
    n = F.shape[0]
    x, iters, setup_time1, solve_time2 = solve_gmres_GNP(C - B @ F, r_vec)
    y = -F @ x
    return np.concatenate((x, y))

def load_matrix(fpath):
    data = np.load(fpath)
    return csr_matrix((data['values'], data['col_indices'], data['row_ptr']), 
                     shape=(data['nrows'][0], data['ncols'][0]))

def main():
    if len(sys.argv) != 6:
        print("Usage: python solver.py <f_path> <b_path> <c_path> <r_path> <s_path>")
        sys.exit(1)
        
    f_path, b_path, c_path, r_path, s_path = sys.argv[1:6]
    
    # Загрузка данных
    F = load_matrix(f_path)
    C = load_matrix(c_path)
    B = load_matrix(b_path)
    r_vec = np.load(r_path)['rhs']
    
    # Решение системы напрямую через блоки
    x = dimension_reduction_solve(F, C, B, r_vec)
    
    # Сохранение результата
    np.savez(s_path, solution=x)
    
if __name__ == '__main__':
    main()