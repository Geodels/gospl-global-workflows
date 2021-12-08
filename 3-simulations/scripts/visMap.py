import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from time import clock

from cartopy import config
import cartopy.crs as ccrs

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from scripts import readOutput as output
import cmocean as cmo

label_size = 7
matplotlib.rcParams["xtick.labelsize"] = label_size
matplotlib.rcParams["ytick.labelsize"] = label_size
matplotlib.rc("font", size=6)

import seaborn as sns
import pandas as pd

import math
import scipy.optimize
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def plotData(
    lons,
    lats,
    norm,
    elev,
    cnorm,
    levels,
    figdir,
    figname,
    color,
    size=(9, 9),
    view=True,
):

    matplotlib.rc("font", size=7)
    data_crs = ccrs.PlateCarree()
    fig = plt.figure(figsize=size)

    data_crs = ccrs.PlateCarree()
    ax = plt.axes(projection=ccrs.Robinson())

    ax.gridlines(draw_labels=True, linewidth=1, color="k", alpha=0.5, linestyle="--")

    contours = ax.contourf(
        lons, lats, norm, levels=levels, norm=cnorm, transform=data_crs, cmap=color
    )

    ax.contour(
        lons, lats, elev, levels=[0], transform=data_crs, colors="k", linewidths=0.5
    )

    ax.set_global()
    ax.autoscale_view()
    plt.colorbar(contours, ax=ax, orientation="horizontal", pad=0.05, shrink=0.5)
    plt.tight_layout()

    fig.savefig(
        figdir + "/pdf/" + figname + "Ma.pdf",
        format="pdf",
        dpi=1200,
        bbox_inches="tight",
    )
    fig.savefig(
        figdir + "/png/" + figname + "Ma.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
    )

    if view:
        plt.show()
    else:
        plt.close(fig)

    return


def computeNorms(map1, map2):
    # Manhattan distance is a distance metric between two points in a N dimensional vector space.
    # It is the sum of the lengths of the projections of the line segment between the points onto the
    # coordinate axes. In simple terms, it is the sum of absolute difference between the measures in
    # all dimensions of two points.

    # Normalize to compensate for exposure difference
    rng = map1.max() - map1.min()
    amin = map1.min()
    map1 = (map1 - amin) * 255 / rng
    rng = map2.max() - map2.min()
    amin = map2.min()
    map2 = (map2 - amin) * 255 / rng

    # Calculate the difference and its norms
    diff = map2 - map1  # Elementwise for scipy arrays

    # Manhattan norm
    n_m = sum(abs(diff))
    # # Zero norm
    # self.n_0 = norm(diff.ravel(), 0)

    # print("  + Manhattan norm: {} / per pixel: {}".format(n_m, n_m/map1.size))
    # print("  + Zero norm: {} / per pixel: {}".format(self.n_0, self.n_0*1.0/map1.size))

    return n_m


def similarity(X, Y=None, *, normalise=True, demean=True):
    """
    Compute similarity between the columns of one or two matrices.

    Covariance: normalise=False, demean=True - https://en.wikipedia.org/wiki/Covariance

    Corrcoef: normalise=True, demean=True - https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient

    Dot product: normalise=False, demean=False

    Cosine similarity: normalise=True, demean=False - https://en.wikipedia.org/wiki/Cosine_similarity
    N.B. also known as the congruence coefficient
    https://en.wikipedia.org/wiki/Congruence_coefficient
    """

    eps = 1.0e-5
    if Y is None:
        if X.ndim != 2:
            raise ValueError("X must be 2D!")
        Y = X

    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must be 2D with the same first dimension!")

    if demean:
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)

    if normalise:
        # Set variances to unity
        x = np.sqrt(np.sum(X ** 2, axis=0))
        x[x < eps] = 1.0
        y = np.sqrt(np.sum(Y ** 2, axis=0))
        y[y < eps] = 1.0
        X = X / x
        Y = Y / y
    else:
        # Or just divide by no. of observations to make an expectation
        X = X / math.sqrt(X.shape[0])
        Y = Y / math.sqrt(Y.shape[0])

    return X.T @ Y


def rv_coefficient(X, Y, *, normalise=True, demean=True):
    """
    RV coefficient (and related variants) between 2D matrices, columnwise
    https://en.wikipedia.org/wiki/RV_coefficient
    RV normally defined in terms of corrcoefs, but any of the above similarity
    metrics will work
    """
    # Based on scalar summaries of covariance matrices
    # Sxy = sim(X,Y)
    # covv_xy = Tr(Sxy @ Syx)
    # rv_xy =  covv_xy / sqrt(covv_xx * covv_yy)

    # Calculate correlations
    # N.B. trace(Xt @ Y) = sum(X * Y)
    Sxy = similarity(X, Y, normalise=normalise, demean=demean)
    c_xy = np.sum(Sxy ** 2)
    Sxx = similarity(X, X, normalise=normalise, demean=demean)
    c_xx = np.sum(Sxx ** 2)
    Syy = similarity(Y, Y, normalise=normalise, demean=demean)
    c_yy = np.sum(Syy ** 2)

    # And put together
    rv = c_xy / math.sqrt(c_xx * c_yy)

    return rv


def concordance_correlation_coefficient(
    y_true, y_pred, sample_weight=None, multioutput="uniform_average"
):
    """Concordance correlation coefficient.
    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    --------
    >>> from sklearn.metrics import concordance_correlation_coefficient
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> concordance_correlation_coefficient(y_true, y_pred)
    0.97678916827853024
    """
    cor = np.corrcoef(y_true, y_pred)[0][1]

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator


def nonMatching(diff, elevrange=100.0):

    nomatch = len(np.where(abs(diff) > elevrange)[0]) / diff.size

    return nomatch


def accuracyScores(obs, sim, elevrange=100.0):

    regions = ["global", "lands", "shelf", "ocean"]

    landIDs = np.where(obs >= 0.0)
    oceanIDs = np.where(obs <= -1000.0)
    shelfIDs = np.where(np.logical_and(obs < 0.0, obs > -1000.0))

    L1 = []
    CV = []
    CR = []
    DP = []
    CS = []
    P = []
    R2 = []
    MAE = []
    RMSE = []
    LCC = []
    NN = []

    # Global
    L1.append(computeNorms(obs.ravel(), sim.ravel()))
    CV.append(rv_coefficient(obs, sim, normalise=False, demean=True))
    CR.append(rv_coefficient(obs, sim, normalise=True, demean=True))
    DP.append(rv_coefficient(obs, sim, normalise=False, demean=False))
    CS.append(rv_coefficient(obs, sim, normalise=True, demean=False))
    P.append(pearsonr(obs.ravel(), sim.ravel())[0])
    R2.append(r2_score(obs.ravel(), sim.ravel(), multioutput="variance_weighted"))
    MAE.append(
        mean_absolute_error(obs.ravel(), sim.ravel(), multioutput="uniform_average")
    )
    RMSE.append(np.sqrt(mean_squared_error(obs.ravel(), sim.ravel())))
    LCC.append(concordance_correlation_coefficient(obs.ravel(), sim.ravel()))
    NN.append(nonMatching(sim - obs, elevrange=elevrange))

    # Land
    Lobs = obs.copy()
    Lsim = sim.copy()
    Lobs[obs < 0] = 0.0
    Lsim[obs < 0] = 0.0
    land_obs = obs[landIDs].ravel()
    land_sim = sim[landIDs].ravel()
    CV.append(rv_coefficient(Lobs, Lsim, normalise=False, demean=True))
    CR.append(rv_coefficient(Lobs, Lsim, normalise=True, demean=True))
    DP.append(rv_coefficient(Lobs, Lsim, normalise=False, demean=False))
    CS.append(rv_coefficient(Lobs, Lsim, normalise=True, demean=False))
    P.append(pearsonr(land_obs.ravel(), land_sim.ravel())[0])
    L1.append(computeNorms(land_obs, land_sim))
    R2.append(r2_score(land_obs, land_sim, multioutput="variance_weighted"))
    MAE.append(mean_absolute_error(land_obs, land_sim, multioutput="uniform_average"))
    RMSE.append(np.sqrt(mean_squared_error(land_obs, land_sim)))
    LCC.append(concordance_correlation_coefficient(land_obs, land_sim))
    NN.append(nonMatching(land_sim - land_obs, elevrange=elevrange))

    # Shelf
    Sobs = obs.copy()
    Ssim = sim.copy()
    Sobs[obs > 0] = 0.0
    Ssim[obs > 0] = 0.0
    Sobs[obs < -1000.0] = 0.0
    Ssim[obs < -1000.0] = 0.0
    shelf_obs = obs[shelfIDs].ravel()
    shelf_sim = sim[shelfIDs].ravel()
    CV.append(rv_coefficient(Sobs, Ssim, normalise=False, demean=True))
    CR.append(rv_coefficient(Sobs, Ssim, normalise=True, demean=True))
    DP.append(rv_coefficient(Sobs, Ssim, normalise=False, demean=False))
    CS.append(rv_coefficient(Sobs, Ssim, normalise=True, demean=False))
    L1.append(computeNorms(shelf_obs, shelf_sim))
    P.append(pearsonr(shelf_obs, shelf_sim)[0])
    R2.append(r2_score(shelf_obs, shelf_sim, multioutput="variance_weighted"))
    MAE.append(mean_absolute_error(shelf_obs, shelf_sim, multioutput="uniform_average"))
    RMSE.append(np.sqrt(mean_squared_error(shelf_obs, shelf_sim)))
    LCC.append(concordance_correlation_coefficient(shelf_obs, shelf_sim))
    NN.append(nonMatching(shelf_sim - shelf_obs, elevrange=elevrange))

    # Ocean
    Oobs = obs.copy()
    Osim = sim.copy()
    Lobs[obs >= -1000.0] = 0.0
    Lsim[obs >= -1000.0] = 0.0
    ocean_obs = obs[oceanIDs].ravel()
    ocean_sim = sim[oceanIDs].ravel()
    CV.append(rv_coefficient(Oobs, Osim, normalise=False, demean=True))
    CR.append(rv_coefficient(Oobs, Osim, normalise=True, demean=True))
    DP.append(rv_coefficient(Oobs, Osim, normalise=False, demean=False))
    CS.append(rv_coefficient(Oobs, Osim, normalise=True, demean=False))
    L1.append(computeNorms(ocean_obs, ocean_sim))
    P.append(pearsonr(ocean_obs.ravel(), ocean_sim.ravel())[0])
    R2.append(
        r2_score(ocean_obs.ravel(), ocean_sim.ravel(), multioutput="variance_weighted")
    )
    MAE.append(
        mean_absolute_error(
            ocean_obs.ravel(), ocean_sim.ravel(), multioutput="uniform_average"
        )
    )
    RMSE.append(np.sqrt(mean_squared_error(ocean_obs.ravel(), ocean_sim.ravel())))
    LCC.append(
        concordance_correlation_coefficient(ocean_obs.ravel(), ocean_sim.ravel())
    )
    NN.append(nonMatching(ocean_sim - ocean_obs, elevrange=elevrange))

    simil = pd.DataFrame(
        {
            "region": regions,
            "covariance": CV,
            "corrcoef": CR,
            "dotproduct": DP,
            "cosine": CS,
            "pearson": P,
            "L1": L1,
        }
    )

    accu = pd.DataFrame(
        {
            "region": regions,
            "RMSE": RMSE,
            "R2": R2,
            "MAE": MAE,
            "LCC": LCC,
            "nonmatching": NN,
        }
    )

    return simil, accu


# def processData(
#     k,
#     figdir,
#     modelpath,
#     modelinput,
#     modelbackward,
#     steps,
#     res=1.0,
#     back=False,
#     view=False,
#     uplift=True,
# ):

#     # t0 = clock()
#     print("-----------------------")
#     if back:
#         # Forward model
#         out = output.readOutput(
#             path=modelpath, filename=modelinput, step=steps[k], uplift=uplift
#         )
#     else:
#         # Forward model
#         out = output.readOutput(
#             path=modelpath, filename=modelinput, step=k, uplift=uplift
#         )

#     out.buildLonLatMesh(res=res, nghb=3)
#     elev = out.z
#     ero = out.th
#     rain = out.rain
#     lats = out.lat - 90.0
#     lons = out.lon - 180.0

#     df_sim = pd.DataFrame(columns=["elev", "erodep", "rain"])
#     df_sim["elev"] = elev.flatten()
#     df_sim["erodep"] = ero.flatten()
#     df_sim["rain"] = rain.flatten()

#     # Backward model
#     if back:
#         out2 = output.readOutput(path=modelpath, filename=modelbackward, step=steps[k])
#         out2.buildLonLatMesh(res=res, nghb=3)
#         elev2 = out2.z

#         # Normalised
#         diff = elev - elev2
#         erodnorm = ero / (ero.max())
#         elevnorm = -elev / (elev.min())
#         diffnorm = diff / (max(abs(diff.min()), diff.max()))

#     if figdir is not None:
#         # Get erosion figure
#         levels = [
#             -0.6,
#             -0.4,
#             -0.2,
#             -0.15,
#             -0.1,
#             -0.05,
#             -0.01,
#             0.01,
#             0.05,
#             0.1,
#             0.15,
#             0.2,
#             0.4,
#             0.6,
#         ]

#         cnorm = colors.Normalize(vmin=levels[0], vmax=levels[-1])
#         levels = [
#             -1.0,
#             -0.9,
#             -0.4,
#             -0.3,
#             -0.2,
#             -0.15,
#             -0.1,
#             -0.05,
#             -0.01,
#             0.01,
#             0.05,
#             0.1,
#             0.15,
#             0.2,
#             0.3,
#             0.4,
#             0.9,
#             1.0,
#         ]

#         figname = "ero-step" + str("{:03d}".format(steps[k]))
#         print("")
#         print("Make erosion figure: ", figname)
#         plotData(
#             lons,
#             lats,
#             erodnorm,
#             elev,
#             cnorm,
#             levels,
#             figdir,
#             figname,
#             color="RdBu_r",
#             size=(9, 9),
#             view=view,
#         )

#         # Get elevation figure
#         levels = [-1.0, -0.8, -0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0]
#         cnorm = colors.Normalize(vmin=levels[0], vmax=levels[-1])

#         levels = [
#             -1.0,
#             -0.9,
#             -0.4,
#             -0.3,
#             -0.2,
#             -0.15,
#             -0.1,
#             -0.05,
#             -0.01,
#             0.01,
#             0.05,
#             0.1,
#             0.15,
#             0.2,
#             0.3,
#             0.4,
#             0.9,
#             1.0,
#         ]

#         figname = "elev-step" + str("{:03d}".format(steps[k]))
#         print("Make elevation figure: ", figname)
#         plotData(
#             lons,
#             lats,
#             elevnorm,
#             elev,
#             cnorm,
#             levels,
#             figdir,
#             figname,
#             color=cmo.cm.delta,
#             size=(9, 9),
#             view=view,
#         )

#         # Get difference figure
#         levels = [-0.1, -0.05, -0.025, -0.01, -0.005, 0.005, 0.01, 0.025, 0.05, 0.1]
#         cnorm = colors.Normalize(vmin=levels[0], vmax=levels[-1])

#         levels = [
#             -1.0,
#             -0.9,
#             -0.4,
#             -0.3,
#             -0.2,
#             -0.15,
#             -0.1,
#             -0.05,
#             -0.01,
#             0.01,
#             0.05,
#             0.1,
#             0.15,
#             0.2,
#             0.3,
#             0.4,
#             0.9,
#             1.0,
#         ]

#         figname = "diff-step" + str("{:03d}".format(steps[k]))
#         print("Make elevation figure: ", figname)
#         print("")
#         plotData(
#             lons,
#             lats,
#             diffnorm,
#             elev,
#             cnorm,
#             levels,
#             figdir,
#             figname,
#             color="BrBG",
#             size=(9, 9),
#             view=view,
#         )

#     if back:
#         # Define pandas dataframe
#         flat_e = elev.flatten()
#         flat_e2 = elev2.flatten()
#         ids = np.where(flat_e2 > -2000.0)[0]

#         val = np.zeros(2 * len(ids))
#         val[: len(ids)] = flat_e[ids]
#         val[len(ids) :] = flat_e2[ids]

#         df_elev = pd.DataFrame(columns=["elev", "Data", "y"])
#         df_elev["elev"] = val
#         df_elev["Data"] = "Model output"
#         strArray = np.array(["Model output" for _ in range(len(val))])
#         strArray[len(ids) :] = "Paleo map"
#         df_elev["Data"] = strArray
#         df_elev["y"] = " "
#         error = diffnorm.flatten()
#         df_2 = pd.DataFrame(
#             columns=["elev", "error", "prediction", "ero", "depo", "erodep", "region"]
#         )
#         df_2["elev"] = flat_e2.flatten()
#         df_2["error"] = np.abs(error * 100.0)
#         over = np.where(error > 0)[0]

#         strArray = np.array(["underestimate" for _ in range(len(flat_e2))])
#         strArray[over] = "overestimate"
#         df_2["prediction"] = strArray

#         erosion = -ero.flatten()
#         erosion[erosion < 0] = 0
#         df_2["ero"] = erosion

#         deposition = ero.flatten()
#         deposition[deposition < 0] = 0
#         df_2["depo"] = deposition

#         df_2["erodep"] = ero.flatten()

#         deep = np.where(flat_e2 < -1000)[0]
#         shelf = np.where(np.logical_and(flat_e2 < 0, flat_e2 >= -1000))[0]
#         land = np.where(np.logical_and(flat_e2 >= 0, flat_e2 <= 500))[0]
#         mountain = np.where(flat_e2 > 500)[0]

#         strArray = np.array(["mountain" for _ in range(len(flat_e2))])
#         strArray[shelf] = "shelf"
#         strArray[land] = "land"
#         strArray[deep] = "deep"
#         df_2["region"] = strArray

#         # Score step
#         simil, accu = accuracyScores(elev2, elev)

#     if back:

#         # print("Processing step {} took {}s".format(steps[k], int(clock() - t0)))
#         return df_sim, df_elev, df_2, simil, accu
#     else:

#         # print("Processing step {} took {}s".format(k, int(clock() - t0)))
#         return df_sim
