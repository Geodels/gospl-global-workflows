import h5py
import glob
import numpy as np
import xarray as xr
from scipy import spatial
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean as cmo
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


class getTectonicInfo:
    def __init__(self, npdata=None, outputDir=None, dt=None, res=0.1):

        self.npdata = npdata
        self.outputDir = outputDir
        self.res = res
        self.dt = dt
        self.ids = None
        self.indices = None
        self.rindices = None

        return

    def xyz2lonlat(self, vertices, radius=6378137.0):

        r = np.sqrt(vertices[:, 0] ** 2 + vertices[:, 1] ** 2 + vertices[:, 2] ** 2)

        xs = np.array(vertices[:, 0])
        ys = np.array(vertices[:, 1])
        zs = np.array(vertices[:, 2] / r)

        lons = np.arctan2(ys, xs)
        lats = np.arcsin(zs)

        # Convert spherical mesh longitudes and latitudes to degrees
        lonlat = np.empty((len(vertices[:, 0]), 2))
        lonlat[:, 0] = np.mod(np.degrees(lons) + 180.0, 360.0) - 180.0
        lonlat[:, 1] = np.mod(np.degrees(lats) + 90, 180.0) - 90.0
        id1 = np.where(lonlat[:, 0] < 0)[0]
        id2 = np.where(lonlat[:, 0] >= 0)[0]
        lonlat[id1, 0] += 180.0
        lonlat[id2, 0] -= 180.0

        return lonlat

    def readGosplData(self, step):

        nbCPUs = len(glob.glob1(self.outputDir + "/h5/", "topology.p*"))

        if nbCPUs == 0:
            nbCPUs = 1

        for k in range(nbCPUs):

            if self.indices is None:
                df = h5py.File("%s/h5/topology.p%s.h5" % (self.outputDir, k), "r")
                coords = np.array((df["/coords"]))

            df2 = h5py.File("%s/h5/gospl.%s.p%s.h5" % (self.outputDir, step, k), "r")
            elev = np.array((df2["/elev"]))
            tec = np.array((df2["/paleotec"]))
            uplift = np.array((df2["/uplift"]))

            if k == 0:
                if self.indices is None:
                    x, y, z = np.hsplit(coords, 3)
                nelev = elev
                ntec = tec
                nuplift = uplift
            else:
                if self.indices is None:
                    x = np.append(x, coords[:, 0])
                    y = np.append(y, coords[:, 1])
                    z = np.append(z, coords[:, 2])
                nelev = np.append(nelev, elev)
                ntec = np.append(ntec, tec)
                nuplift = np.append(nuplift, uplift)

            if self.indices is None:
                df.close()
            df2.close()

        if self.indices is None:
            nbPts = len(x)
            ncoords = np.zeros((nbPts, 3))
            ncoords[:, 0] = x.ravel()
            ncoords[:, 1] = y.ravel()
            ncoords[:, 2] = z.ravel()

            # Load mesh structure
            mesh_struct = np.load(str(self.npdata))
            self.vertices = mesh_struct["v"]
            self.cells = mesh_struct["c"]
            self.lonlat = self.xyz2lonlat(self.vertices)

            tree = spatial.cKDTree(ncoords, leafsize=10)
            distances, self.indices = tree.query(self.vertices, k=1)

        self.elev = nelev[self.indices]
        self.tec = ntec[self.indices]
        self.uplift = nuplift[self.indices]

        return

    def mapRegular(self, step):

        self.readGosplData(step)

        if self.rindices is None:
            self.nx = int(360.0 / self.res) + 1
            self.ny = int(180.0 / self.res) + 1
            self.lon = np.linspace(-180.0, 180.0, self.nx)
            self.lat = np.linspace(-90.0, 90.0, self.ny)
            lonv, latv = np.meshgrid(self.lon, self.lat)
            rlonlat = np.dstack([lonv.flatten(), latv.flatten()])[0]
            nghb = 3
            self.rtree = spatial.cKDTree(rlonlat, leafsize=10)
            tree = spatial.cKDTree(self.lonlat, leafsize=10)
            distances, self.rindices = tree.query(rlonlat, k=nghb)
            self.onIDs = np.where(distances[:, 0] == 0)[0]
            distances[self.onIDs, :] = 0.001
            self.weights = 1.0 / distances ** 2
            self.denum = 1.0 / np.sum(self.weights, axis=1)
            self.denum[self.onIDs] = 0.0

        zi = np.sum(self.weights * self.elev[self.rindices], axis=1) * self.denum
        teci = np.sum(self.weights * self.uplift[self.rindices], axis=1) * self.denum
        paleoteci = np.sum(self.weights * self.tec[self.rindices], axis=1) * self.denum

        if len(self.onIDs) > 0:
            zi[self.onIDs] = self.elev[self.rindices[self.onIDs, 0]]
            teci[self.onIDs] = self.uplift[self.rindices[self.onIDs, 0]]
            paleoteci[self.onIDs] = self.tec[self.rindices[self.onIDs, 0]]

        rpaleotec = np.reshape(paleoteci, (self.ny, self.nx))
        rtec = np.reshape(teci, (self.ny, self.nx))
        relev = np.reshape(zi, (self.ny, self.nx))

        self.zds = xr.DataArray(
            data=relev, dims=["lat", "lon"], coords={"lon": self.lon, "lat": self.lat,},
        )

        self.tecds = xr.DataArray(
            data=rtec * self.dt,
            dims=["lat", "lon"],
            coords={"lon": self.lon, "lat": self.lat,},
        )

        self.paleotecds = xr.DataArray(
            data=rpaleotec,
            dims=["lat", "lon"],
            coords={"lon": self.lon, "lat": self.lat,},
        )

        return

    def updatePaleoTec(self, smth=8, factor=1.5):

        lon = self.zds.lon.values
        lat = self.zds.lat.values
        ptec = self.paleotecds.values.copy()
        relev = self.zds.values.copy()
        nptec = gaussian_filter(ptec, sigma=smth)

        upIDs = np.where((ptec >= 0) & (relev > 0))
        ptec[upIDs] = nptec[upIDs] * factor

        self.newtecds = xr.DataArray(
            data=ptec, dims=["lat", "lon"], coords={"lon": lon, "lat": lat,},
        )

        return

    def remapTec(self):

        if self.ids is None:
            nghb = 3
            distances, self.ids = self.rtree.query(self.lonlat, k=nghb)
            self.ronIDs = np.where(distances[:, 0] == 0)[0]
            distances[self.ronIDs, :] = 0.001
            self.rweights = 1.0 / distances ** 2
            self.rdenum = 1.0 / np.sum(self.rweights, axis=1)
            self.rdenum[self.ronIDs] = 0.0

        tec = self.newtecds.values.flatten()
        ntec = np.sum(self.rweights * tec[self.ids], axis=1) * self.rdenum
        if len(self.ronIDs) > 0:
            ntec[self.ronIDs] = tec[self.ids[self.ronIDs, 0]]

        self.ntec = np.zeros(len(self.tec))
        ids = np.where((ntec > 0.0) & (self.elev > 0.0))[0]
        self.ntec[ids] = ntec[ids]
        ids = np.where(self.tec <= 0.0)[0]
        self.ntec[ids] = self.tec[ids]

        return

    def plotTopo(self, minh=-5000, maxh=5000, Robinson=False):

        if Robinson:
            subplot_kws = dict(projection=ccrs.Robinson(), facecolor="white")
        else:
            subplot_kws = dict(projection=ccrs.PlateCarree(), facecolor="white")
        fig = plt.figure(figsize=[18, 14])
        if Robinson:
            ax = fig.add_subplot(projection=ccrs.Robinson())
        else:
            ax = fig.add_subplot(projection=ccrs.PlateCarree())

        # ax.set_title("SIMULATION STEP: " + str(step), loc="left", fontsize=15)

        p = self.zds.plot(
            x="lon",
            y="lat",
            vmin=minh,
            vmax=maxh,
            cmap=cmo.cm.topo,
            subplot_kws=subplot_kws,
            transform=ccrs.PlateCarree(),
            add_labels=False,
            add_colorbar=False,
        )

        self.zds.plot.contour(
            ax=ax,
            vmin=0,
            vmax=0.1,
            levels=2,
            linewidths=0.5,
            colors="white",
            add_colorbar=False,
            transform=ccrs.PlateCarree(),
            extend="neither",
            add_labels=False,
        )

        # add separate colorbar
        cb = plt.colorbar(p, shrink=0.5)
        cb.ax.tick_params(labelsize=10)
        cb.ax.get_yaxis().labelpad = 15
        cb.ax.set_ylabel("Elevation (m)", rotation=270, fontsize=13)

        # draw parallels/meridians and write labels
        gl = p.axes.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color="white",
            alpha=0.5,
            linestyle="--",
        )

        # adjust labels to taste
        if not Robinson:
            gl.top_labels = False
            gl.right_labels = False
            gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {"size": 10, "color": "black"}
            gl.ylabel_style = {"size": 10, "color": "black"}
        plt.show()

        return

    def plotTecto(self, tecds, minu=-1000, maxu=1000, Robinson=False):

        if Robinson:
            subplot_kws = dict(projection=ccrs.Robinson(), facecolor="white")
        else:
            subplot_kws = dict(projection=ccrs.PlateCarree(), facecolor="white")
        fig = plt.figure(figsize=[18, 14])
        if Robinson:
            ax = fig.add_subplot(projection=ccrs.Robinson())
        else:
            ax = fig.add_subplot(projection=ccrs.PlateCarree())

        # ax.set_title("SIMULATION STEP: " + str(step), loc="left", fontsize=15)

        p = tecds.plot(
            x="lon",
            y="lat",
            vmin=minu,
            vmax=maxu,
            cmap="seismic",
            subplot_kws=subplot_kws,
            transform=ccrs.PlateCarree(),
            add_labels=False,
            add_colorbar=False,
        )

        self.zds.plot.contour(
            ax=ax,
            vmin=0,
            vmax=0.1,
            levels=2,
            linewidths=0.5,
            colors="white",
            add_colorbar=False,
            transform=ccrs.PlateCarree(),
            extend="neither",
            add_labels=False,
        )

        # add separate colorbar
        cb = plt.colorbar(p, shrink=0.5)
        cb.ax.tick_params(labelsize=10)
        cb.ax.get_yaxis().labelpad = 15
        cb.ax.set_ylabel(
            "Tectonic forcing from paleo-elevation (m)", rotation=270, fontsize=13
        )

        # draw parallels/meridians and write labels
        gl = p.axes.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color="black",
            alpha=0.5,
            linestyle="--",
        )

        # adjust labels to taste
        if not Robinson:
            gl.top_labels = False
            gl.right_labels = False
            gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {"size": 10, "color": "black"}
            gl.ylabel_style = {"size": 10, "color": "black"}
        plt.show()

        return

    def plotTectoSPM(self, paleotecds, minu=-500, maxu=500, Robinson=False):

        if Robinson:
            subplot_kws = dict(projection=ccrs.Robinson(), facecolor="white")
        else:
            subplot_kws = dict(projection=ccrs.PlateCarree(), facecolor="white")
        fig = plt.figure(figsize=[18, 14])
        if Robinson:
            ax = fig.add_subplot(projection=ccrs.Robinson())
        else:
            ax = fig.add_subplot(projection=ccrs.PlateCarree())

        # ax.set_title("SIMULATION STEP: " + str(step), loc="left", fontsize=15)

        p = paleotecds.plot(
            x="lon",
            y="lat",
            vmin=minu,
            vmax=maxu,
            cmap="seismic",
            subplot_kws=subplot_kws,
            transform=ccrs.PlateCarree(),
            add_labels=False,
            add_colorbar=False,
        )

        self.zds.plot.contour(
            ax=ax,
            vmin=0,
            vmax=0.1,
            levels=2,
            linewidths=0.5,
            colors="black",
            add_colorbar=False,
            transform=ccrs.PlateCarree(),
            extend="neither",
            add_labels=False,
        )

        # add separate colorbar
        cb = plt.colorbar(
            p,  # ticks=[-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500],
            shrink=0.5,
        )
        cb.ax.tick_params(labelsize=10)
        cb.ax.get_yaxis().labelpad = 15
        cb.ax.set_ylabel(
            "Erosion/deposition induced vertical change (m)", rotation=270, fontsize=13
        )

        # draw parallels/meridians and write labels
        gl = p.axes.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color="black",
            alpha=0.5,
            linestyle="--",
        )

        # adjust labels to taste
        if not Robinson:
            gl.top_labels = False
            gl.right_labels = False
            gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {"size": 10, "color": "black"}
            gl.ylabel_style = {"size": 10, "color": "black"}
        plt.show()
