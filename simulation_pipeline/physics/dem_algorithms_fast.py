import numpy as np
import cupy as cp
from cupyx.scipy.linalg import lu_factor, lu_solve
import numba


def simple_reg_dem_gpu(data, errors, exptimes, logt, tresps,
                       kmax=100, kcon=5, steps=[0.1, 0.5], drv_con=8.0, chi2_th=1.0, tol=0.1):
    """
    Computes a DEM given a set of input data, errors, exposure times, temperatures, and
    temperature response functions.

    Author: Joseph Plowman -- 09-15-2021
    See: https://ui.adsabs.harvard.edu/abs/2020ApJ...905...17P/abstract

    Parameters
    ----------
    data: image data (dimensions n_x by n_y by n_channels)
    errors: image of uncertainties in data (dimensions n_x by n_y by n_channels)
    exptimes: exposure times for each image (dimensions n_channels)
    logt: Array of temperatures (dimensions n_temperatures)
    tresps: Arrays of temperature response functions (dimensions n_temperatures by n_channels)
    kmax:
        The maximum number of iteration steps (default - 100).
    kcon:
        Initial number of steps before terminating if chi^2 never improves (default - 5).
    steps:
        Two element array containing large and small step sizes (default - 0.1 and 0.5).
    drv_con:
        The size of the derivative constraint - threshold delta_0 limiting the change in
        log DEM per unit input temperature (default - 8; e.g., per unit log_{10}(T)).
    chi2_th:
        Reduced chi^2 threshold for termination (default - 1).
    tol:
        How close to chi2_th the reduced chi^2 needs to be (default - 0.1).

    Returns
    --------
    dem: (dimensions n_x by n_y by n_temperatures).
        Dimensions depend on the units of logt (need not be log temperature, despite the name) 
        and tresps. For instance, if tresps has units cm^5 and the logt values 
        are base 10 log of temperature in Kelvin, the output DEM will be 
        in units of cm^{-5} per unit Log_{10}(T).
    chi2: Array of reduced chi squared values (dimensions n_x by n_y)
    """
    data = cp.array(data)
    exptimes = cp.array(exptimes)
    tresps = cp.array(tresps)

    [nt, nd] = tresps.shape
    nt_ones = cp.ones(nt)
    [nx, ny, nd] = data.shape
    dT = logt[1:nt]-logt[0:nt-1]
    [dTleft, dTright] = [cp.diag(cp.hstack([dT, 0])), cp.diag(cp.hstack([0, dT]))]
    [idTleft, idTright] = [cp.diag(cp.hstack([1.0/dT, 0])), cp.diag(cp.hstack([0, 1.0/dT]))]
    Bij = ((dTleft+dTright)*2.0 + cp.roll(dTright, -1, axis=0) + cp.roll(dTleft, 1, axis=0))/6.0
    # Matrix mapping coefficents to data
    Rij = cp.matmul((tresps*cp.outer(nt_ones, exptimes)).T, Bij)
    Dij = idTleft+idTright - cp.roll(idTright, -1, axis=0) - cp.roll(idTleft, 1, axis=0)
    regmat = Dij*nd/(drv_con**2*(logt[nt-1]-logt[0]))
    rvec = cp.sum(Rij, axis=1)

    dems = cp.zeros([nx, ny, nt])
    chi2 = cp.zeros([nx, ny]) - 1.0
    for i in range(0, nx):
        for j in range(0, ny):
            err = errors[i, j, :]
            dat0 = cp.clip(data[i, j, :], 0.0, None)
            s = cp.log(cp.sum((rvec)*((dat0 > 1.0e-2)/err**2))/cp.sum((rvec/err)**2)/nt_ones)
            for k in range(0, kmax):
                # Correct data by f(s)-s*f'(s)
                dat = (dat0-cp.matmul(Rij, ((1-s)*cp.exp(s)))) / err
                mmat = Rij*cp.outer(1.0/err, cp.exp(s))
                amat = cp.matmul(mmat.T, mmat) + regmat
                try:
                    [c, low] = lu_factor(amat)
                except:
                    break
                c2p = cp.mean((dat0-cp.dot(Rij, cp.exp(s)))**2/err**2)
                deltas = lu_solve((c, low), cp.dot(mmat.T, dat))-s
                deltas *= cp.clip(cp.max(cp.abs(deltas)), None, 0.5/steps[0])/cp.max(cp.abs(deltas))
                ds = 1-2*(c2p < chi2_th)  # Direction sign; is chi squared too large or too small?
                c20 = cp.mean((dat0-cp.dot(Rij, cp.exp(s+deltas*ds*steps[0])))**2.0/err**2.0)
                c21 = cp.mean((dat0-cp.dot(Rij, cp.exp(s+deltas*ds*steps[1])))**2.0/err**2.0)
                interp_step = ((steps[0]*(c21-chi2_th)+steps[1]*(chi2_th-c20))/(c21-c20))
                s += deltas*ds*cp.clip(interp_step, steps[0], steps[1])
                chi2[i, j] = cp.mean((dat0-cp.dot(Rij, cp.exp(s)))**2/err**2)
                condition_1 = (ds*(c2p-c20)/steps[0] < tol)*(k > kcon)
                condition_2 = cp.abs(chi2[i, j]-chi2_th) < tol
                if condition_1 or condition_2:
                    break
            dems[i, j, :] = cp.exp(s)

    dems = cp.asnumpy(dems)
    chi2 = cp.asnumpy(chi2)

    return dems, chi2


@numba.jit(nopython=True)
def simple_reg_dem_numba(data, errors, exptimes, logt, tresps,
                         kmax=100, kcon=5, steps=[0.1, 0.5], drv_con=8.0, chi2_th=1.0, tol=0.1):
    """
    Computes a DEM given a set of input data, errors, exposure times, temperatures, and
    temperature response functions.

    Author: Joseph Plowman -- 09-15-2021
    See: https://ui.adsabs.harvard.edu/abs/2020ApJ...905...17P/abstract

    Parameters
    ----------
    data: image data (dimensions n_x by n_y by n_channels)
    errors: image of uncertainties in data (dimensions n_x by n_y by n_channels)
    exptimes: exposure times for each image (dimensions n_channels)
    logt: Array of temperatures (dimensions n_temperatures)
    tresps: Arrays of temperature response functions (dimensions n_temperatures by n_channels)
    kmax:
        The maximum number of iteration steps (default - 100).
    kcon:
        Initial number of steps before terminating if chi^2 never improves (default - 5).
    steps:
        Two element array containing large and small step sizes (default - 0.1 and 0.5).
    drv_con:
        The size of the derivative constraint - threshold delta_0 limiting the change in
        log DEM per unit input temperature (default - 8; e.g., per unit log_{10}(T)).
    chi2_th:
        Reduced chi^2 threshold for termination (default - 1).
    tol:
        How close to chi2_th the reduced chi^2 needs to be (default - 0.1).

    Returns
    --------
    dem: (dimensions n_x by n_y by n_temperatures).
        Dimensions depend on the units of logt (need not be log temperature, despite the name) 
        and tresps. For instance, if tresps has units cm^5 and the logt values 
        are base 10 log of temperature in Kelvin, the output DEM will be 
        in units of cm^{-5} per unit Log_{10}(T).
    chi2: Array of reduced chi squared values (dimensions n_x by n_y)
    """
    [nt, nd] = tresps.shape
    nt_ones = np.ones(nt)
    [nx, ny, nd] = data.shape
    dT = logt[1:nt]-logt[0:nt-1]
    [dTleft, dTright] = [np.diag(np.hstack([dT, 0])), np.diag(np.hstack([0, dT]))]
    [idTleft, idTright] = [np.diag(np.hstack([1.0/dT, 0])), np.diag(np.hstack([0, 1.0/dT]))]
    Bij = ((dTleft+dTright)*2.0 + np.roll(dTright, -1, axis=0) + np.roll(dTleft, 1, axis=0))/6.0
    # Matrix mapping coefficents to data
    Rij = np.matmul((tresps*np.outer(nt_ones, exptimes)).T, Bij)
    Dij = idTleft+idTright - np.roll(idTright, -1, axis=0) - np.roll(idTleft, 1, axis=0)
    regmat = Dij*nd/(drv_con**2*(logt[nt-1]-logt[0]))
    rvec = np.sum(Rij, axis=1)

    dems = np.zeros([nx, ny, nt])
    chi2 = np.zeros([nx, ny]) - 1.0
    for i in range(0, nx):
        for j in range(0, ny):
            err = errors[i, j, :]
            dat0 = np.clip(data[i, j, :], 0.0, None)
            s = np.log(np.sum((rvec)*((dat0 > 1.0e-2)/err**2))/np.sum((rvec/err)**2)/nt_ones)
            for k in range(0, kmax):
                # Correct data by f(s)-s*f'(s)
                dat = (dat0-np.matmul(Rij, ((1-s)*np.exp(s)))) / err
                mmat = Rij*np.outer(1.0/err, np.exp(s))
                amat = np.matmul(mmat.T, mmat) + regmat
                try:
                    L = np.linalg.cholesky(amat)
                except np.linalg.LinAlgError:
                    break
                b = np.dot(mmat.T, dat)
                deltas = np.linalg.solve(L.T, np.linalg.solve(L, b)) - s
                c2p = np.mean((dat0-np.dot(Rij, np.exp(s)))**2/err**2)
                deltas *= np.clip(np.max(np.abs(deltas)), None, 0.5/steps[0])/np.max(np.abs(deltas))
                ds = 1-2*(c2p < chi2_th)  # Direction sign; is chi squared too large or too small?
                c20 = np.mean((dat0-np.dot(Rij, np.exp(s+deltas*ds*steps[0])))**2.0/err**2.0)
                c21 = np.mean((dat0-np.dot(Rij, np.exp(s+deltas*ds*steps[1])))**2.0/err**2.0)
                interp_step = ((steps[0]*(c21-chi2_th)+steps[1]*(chi2_th-c20))/(c21-c20))
                s += deltas*ds*np.clip(interp_step, steps[0], steps[1])
                chi2[i, j] = np.mean((dat0-np.dot(Rij, np.exp(s)))**2/err**2)
                condition_1 = (ds*(c2p-c20)/steps[0] < tol)*(k > kcon)
                condition_2 = np.abs(chi2[i, j]-chi2_th) < tol
                if condition_1 or condition_2:
                    break
            dems[i, j, :] = np.exp(s)

    return dems, chi2
