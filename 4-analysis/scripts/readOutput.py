import os
import gc
import sys
import glob
import h5py
import time
import meshio
import numpy as np
import pandas as pd
from scipy import spatial
import ruamel.yaml as yaml
from scipy.interpolate import interp1d

import cftime
from datetime import datetime, timedelta
import netCDF4 as nc
from netCDF4 import num2date, date2num, date2index

import heapq
import numpy as np
from math import sqrt

from scipy import ndimage
from scipy import sparse as sp
from scipy.sparse import linalg as splg
from matplotlib import cm

import richdem as rd

from gospl._fortran import filllabel


class readOutput:
    def __init__(
        self, path=None, filename=None, step=None, nbstep=None, back=False, uplift=True
    ):

        # Check input file exists
        self.path = path
        if path is not None:
            filename = self.path + filename

        try:
            with open(filename) as finput:
                pass
        except IOError:
            print("Unable to open file: ", filename)
            raise IOError("The input file is not found...")

        # Open YAML file
        with open(filename, "r") as finput:
            self.input = yaml.load(finput, Loader=yaml.Loader)

        self.step = step
        self.nbstep = nbstep
        self.radius = 6378137.0
        self.back = back
        self.lookuplift = uplift
        self._inputParser()
        self.nx = None

        self.nbCPUs = len(glob.glob1(self.outputDir + "/h5/", "topology.p*"))

        self._readElevationData(step)

        if self.step == 0 or nbstep is None:
            self.datafA = None
            self.datafSed = None
            self.datafRain = None
            self.datafelev = None
            self.datafUp = None
            self.datafhdisp = None
            self.datafEroDep = None
            self.datafBasin = None

        return

    def update(self, step=None, uplift=True):

        self.step = step
        self.lookuplift = uplift
        self._readElevationData(step)

        return

    def _inputParser(self):

        try:
            domainDict = self.input["domain"]
        except KeyError:
            print("Key 'domain' is required and is missing in the input file!")
            raise KeyError("Key domain is required in the input file!")

        try:
            self.npdata = domainDict["npdata"]
        except KeyError:
            print(
                "Key 'npdata' is required and is missing in the 'domain' declaration!"
            )
            raise KeyError("Simulation npdata needs to be declared.")

        try:
            timeDict = self.input["time"]
        except KeyError:
            print("Key 'time' is required and is missing in the input file!")
            raise KeyError("Key time is required in the input file!")

        try:
            self.tStart = timeDict["start"]
        except KeyError:
            print("Key 'start' is required and is missing in the 'time' declaration!")
            raise KeyError("Simulation start time needs to be declared.")

        try:
            self.tout = timeDict["tout"]
        except KeyError:
            print("Key 'tout' is required and is missing in the 'time' declaration!")
            raise KeyError("Simulation output time needs to be declared.")

        try:
            self.tEnd = timeDict["end"]
        except KeyError:
            print("Key 'end' is required and is missing in the 'time' declaration!")
            raise KeyError("Simulation end time needs to be declared.")

        try:
            outDict = self.input["output"]
            try:
                self.outputDir = outDict["dir"]
            except KeyError:
                self.outputDir = "output"
        except KeyError:
            self.outputDir = "output"

        if self.path is not None:
            self.outputDir = self.path + self.outputDir

        seafile = None
        self.seacurve = False
        self.sealevel = 0.0
        try:
            seaDict = self.input["sea"]
            try:
                self.sealevel = seaDict["position"]
                try:
                    seafile = seaDict["curve"]
                except KeyError:
                    seafile = None
            except KeyError:
                try:
                    seafile = seaDict["curve"]
                except KeyError:
                    seafile = None
        except KeyError:
            self.sealevel = 0.0

        if seafile is not None:
            try:
                with open(seafile) as fsea:
                    fsea.close()
                    try:
                        seadata = pd.read_csv(
                            seafile,
                            sep=r",",
                            engine="c",
                            header=None,
                            na_filter=False,
                            dtype=np.float,
                            low_memory=False,
                        )

                    except ValueError:
                        try:
                            seadata = pd.read_csv(
                                seafile,
                                sep=r"\s+",
                                engine="c",
                                header=None,
                                na_filter=False,
                                dtype=np.float,
                                low_memory=False,
                            )

                        except ValueError:
                            print(
                                "The sea-level file is not well formed: it should be comma or tab separated",
                                flush=True,
                            )
                            raise ValueError("Wrong formating of sea-level file.")
            except IOError:
                print("Unable to open file: ", seafile)
                raise IOError("The sealevel file is not found...")

            self.seacurve = True
            seadata[1] += self.sealevel
            if seadata[0].min() > self.tStart:
                tmpS = []
                tmpS.insert(0, {0: self.tStart, 1: seadata[1].iloc[0]})
                seadata = pd.concat([pd.DataFrame(tmpS), seadata], ignore_index=True)
            if seadata[0].max() < self.tEnd:
                tmpE = []
                tmpE.insert(0, {0: self.tEnd, 1: seadata[1].iloc[-1]})
                seadata = pd.concat([seadata, pd.DataFrame(tmpE)], ignore_index=True)
            self.seafunction = interp1d(seadata[0], seadata[1], kind="linear")

            self.time = np.arange(self.tStart, self.tEnd + 0.1, self.tout)

        return

    def pit_fill(self, data):

        dem = np.copy(data)
        elev = np.copy(data)

        nrow, ncol = dem.shape
        dem[:, :] = 100000.0
        dem[0, :] = elev[0, :]
        dem[-1, :] = elev[-1, :]
        dem[:, 0] = elev[:, 0]
        dem[:, -1] = elev[:, -1]
        flag = True

        # Neighbours along x,y directions
        direct8y = [1, 1, 1, 0, -1, -1, -1, 0]
        direct8x = [-1, 0, 1, 1, 1, 0, -1, -1]

        # Main loop
        while flag == True:
            flag = False
            for i in range(1, nrow - 1):
                for j in range(1, ncol - 1):
                    if elev[i, j] <= 0.0:
                        dem[i, j] = elev[i, j]
                    elif dem[i, j] > elev[i, j]:
                        for p in range(0, 8):
                            r = i + direct8x[p]
                            c = j + direct8y[p]
                            if elev[i, j] >= dem[r, c] + 0.01:
                                dem[i, j] = elev[i, j]
                                flag = True
                            else:
                                if dem[i, j] > dem[r, c] + 0.01:
                                    dem[i, j] = dem[r, c] + 0.01
                                    flag = True
        output_array = dem
        return output_array

    def flow_direction_d8(self, DX, DY, dem, fill_pits=False):

        if fill_pits:
            dem = self.pit_fill(dem)

        nr = dem.shape[0]
        nc = dem.shape[1]

        HYP = sqrt(DX * DX + DY * DY)

        ghost = np.zeros((nr + 2, nc + 2), "d")
        slp = np.copy(ghost)
        ghost[1:-1, 1:-1] = dem[:, :]
        ghost[0, 1:-1] = dem[0, :]
        ghost[-1, 1:-1] = dem[-1, :]
        ghost[1:-1, 0] = dem[:, 0]
        ghost[1:-1, -1] = dem[:, -1]
        ghost[0, 0] = ghost[1, 1]
        ghost[-1, -1] = ghost[-2, -2]
        ghost[0, -1] = ghost[1, -2]
        ghost[-1, 0] = ghost[-2, 1]

        neig_incr_row_major = np.array(
            [1, -nc + 1, -nc, -nc - 1, -1, -1 + nc, nc, nc + 1]
        )
        neig_incr_col_major = np.array(
            [nr, nr - 1, -1, -nr - 1, -nr, -nr + 1, 1, nr + 1]
        )
        neig_incr = neig_incr_col_major

        slopes = np.zeros((8,), "d")

        all_indices = np.arange(nr * nc, dtype="i")
        max_indices = np.zeros(nr * nc, "i")
        slope_vals = np.zeros(nr * nc, "d")
        slope_count = np.zeros(nr * nc, "i")

        for i in range(1, nr + 1):
            for j in range(1, nc + 1):
                slopes[0] = (ghost[i, j] - ghost[i, j + 1]) / DX
                slopes[4] = (ghost[i, j] - ghost[i, j - 1]) / DX
                slopes[1] = (ghost[i, j] - ghost[i - 1, j + 1]) / HYP
                slopes[2] = (ghost[i, j] - ghost[i - 1, j]) / DY
                slopes[3] = (ghost[i, j] - ghost[i - 1, j - 1]) / HYP
                slopes[5] = (ghost[i, j] - ghost[i + 1, j - 1]) / HYP
                slopes[6] = (ghost[i, j] - ghost[i + 1, j]) / DY
                slopes[7] = (ghost[i, j] - ghost[i + 1, j + 1]) / HYP
                glob_ind = (j - 1) * nr + i - 1  # local cell col major
                loc_max = slopes.argmax()
                if slopes[loc_max] > 0:
                    glob_max = min(max(glob_ind + neig_incr[loc_max], 0), nc * nr - 1)
                    max_indices[glob_ind] = glob_max
                    slope_vals[glob_ind] = slopes[loc_max]
                    slope_count[glob_ind] = 1
                    slp[i, j] = slope_vals[glob_ind]
        M = sp.csr_matrix(
            (slope_count, (all_indices, max_indices)), shape=(nr * nc, nr * nc)
        )

        return M, slp[1:-1, 1:-1]

    def flowAccum(self, M, shape):

        nc = M.shape[1]
        I = sp.eye(M.shape[0], M.shape[1])
        B = I - M.transpose()
        b = np.ones(nc, "d")

        fa = splg.spsolve(B, b)
        fa = fa.reshape(shape, order="F")

        return fa

    def drainageBasins(self, M, dsh):

        ntot = dsh[0] * dsh[1]
        assert M.shape[0] == ntot and M.shape[1] == ntot

        # Find cells that receive water only
        routes = M.sum(axis=1)  # 1 if cell routes flow to another cell, 0 otherwise
        receives = M.sum(axis=0)  # number of neighbors that flow to each cell

        # Cells that don't route flow but also receive it
        index = np.logical_and(routes.flat == 0, receives.flat != 0)
        drains = np.zeros(ntot, "d")
        drains[index] = 1

        # Grab cells that are drainage outlets
        outlets = np.where(drains == 1)
        drains = np.cumsum(drains) * drains

        # Solve flow to drains
        I = sp.eye(ntot, ntot)
        A = I - M
        X = splg.spsolve(A, drains)
        index_orph = np.logical_and(routes.flat == 0, receives.flat == 0)
        X[index_orph] = -1
        index_orph = np.logical_not(index_orph)
        x, iix = np.unique(X[index_orph], return_inverse=True)
        X[index_orph] = iix
        X = X.reshape(dsh, order="F")

        return X, outlets

    def lonlat2xyz(self, lon, lat, radius=6378137.0):

        rlon = np.radians(lon)
        rlat = np.radians(lat)

        coords = np.zeros((3))
        coords[0] = np.cos(rlat) * np.cos(rlon) * radius
        coords[1] = np.cos(rlat) * np.sin(rlon) * radius
        coords[2] = np.sin(rlat) * radius

        return coords

    def _xyz2lonlat(self):

        r = np.sqrt(
            self.vertices[:, 0] ** 2
            + self.vertices[:, 1] ** 2
            + self.vertices[:, 2] ** 2
        )
        h = r - self.radius

        xs = np.array(self.vertices[:, 0])
        ys = np.array(self.vertices[:, 1])
        zs = np.array(self.vertices[:, 2] / r)

        lons = np.arctan2(ys, xs)
        lats = np.arcsin(zs)

        # Convert spherical mesh longitudes and latitudes to degrees
        self.lonlat = np.empty((len(self.vertices[:, 0]), 2))
        self.lonlat[:, 0] = np.mod(np.degrees(lons) + 180.0, 360.0) - 180.0
        self.lonlat[:, 1] = np.mod(np.degrees(lats) + 90, 180.0) - 90.0

        self.tree = spatial.cKDTree(self.lonlat, leafsize=10)

        return

    def _getCoordinates(self, step):

        if self.nbCPUs == 0:
            self.nbCPUs = 1

        self.hdisp = None
        self.uplift = None

        for k in range(self.nbCPUs):
            df = h5py.File("%s/h5/topology.p%s.h5" % (self.outputDir, k), "r")
            coords = np.array((df["/coords"]))

            df2 = h5py.File("%s/h5/gospl.%s.p%s.h5" % (self.outputDir, step, k), "r")
            elev = np.array((df2["/elev"]))
            rain = np.array((df2["/rain"]))
            if not self.back:
                erodep = np.array((df2["/erodep"]))
                sedLoad = np.array((df2["/sedLoad"]))
                flowAcc = np.array((df2["/fillFA"]))
                if self.lookuplift and step > 0:
                    uplift = np.array((df2["/uplift"]))
                    # hdisp = np.array((df2["/hdisp"]))

            if self.seacurve:
                sealevel = self.seafunction(self.time[step])
                elev -= sealevel
            else:
                elev -= self.sealevel

            if k == 0:
                x, y, z = np.hsplit(coords, 3)
                nelev = elev
                nrain = rain

                if not self.back:
                    nerodep = erodep
                    nsedLoad = sedLoad
                    nflowAcc = flowAcc
                    if self.lookuplift and step > 0:
                        # nhdisp = hdisp
                        nuplift = uplift
            else:
                x = np.append(x, coords[:, 0])
                y = np.append(y, coords[:, 1])
                z = np.append(z, coords[:, 2])
                nelev = np.append(nelev, elev)
                nrain = np.append(nrain, rain)

                if not self.back:
                    nerodep = np.append(nerodep, erodep)
                    nsedLoad = np.append(nsedLoad, sedLoad)
                    nflowAcc = np.append(nflowAcc, flowAcc)
                    if self.lookuplift and step > 0:
                        # nhdisp = np.append(nhdisp, hdisp)
                        nuplift = np.append(nuplift, uplift)

            df.close()

        self.nbPts = len(x)
        ncoords = np.zeros((self.nbPts, 3))
        ncoords[:, 0] = x.ravel()
        ncoords[:, 1] = y.ravel()
        ncoords[:, 2] = z.ravel()
        if not self.back:
            if self.lookuplift and step > 0:
                del coords, elev, erodep, uplift, sedLoad, flowAcc, rain
            elif self.lookuplift and step == 0:
                del coords, elev, erodep, sedLoad, flowAcc, rain
            else:
                del coords, elev
        else:
            del coords, elev
        gc.collect()

        # Load mesh structure
        mesh_struct = np.load(str(self.npdata) + ".npz")
        self.vertices = mesh_struct["v"]
        self.cells = mesh_struct["c"]
        self.ngbID = mesh_struct["n"]
        self._xyz2lonlat()

        tree = spatial.cKDTree(ncoords, leafsize=10)
        distances, indices = tree.query(self.vertices, k=3)

        # Inverse weighting distance...
        weights = 1.0 / distances ** 2
        onIDs = np.where(distances[:, 0] == 0)[0]
        if nelev[indices].ndim == 2:
            self.elev = np.sum(weights * nelev[indices][:, :], axis=1) / np.sum(
                weights, axis=1
            )
            self.rain = np.sum(weights * nrain[indices][:, :], axis=1) / np.sum(
                weights, axis=1
            )

            if not self.back:
                self.erodep = np.sum(weights * nerodep[indices][:, :], axis=1) / np.sum(
                    weights, axis=1
                )
                self.sedLoad = np.sum(
                    weights * nsedLoad[indices][:, :], axis=1
                ) / np.sum(weights, axis=1)
                self.flowAcc = np.sum(
                    weights * nflowAcc[indices][:, :], axis=1
                ) / np.sum(weights, axis=1)
                if self.lookuplift and step > 0:
                    # self.hdisp = np.sum(
                    #     weights * nhdisp[indices][:, :], axis=1
                    # ) / np.sum(weights, axis=1)
                    self.uplift = np.sum(
                        weights * nuplift[indices][:, :], axis=1
                    ) / np.sum(weights, axis=1)

        else:
            self.elev = np.sum(weights * nelev[indices][:, :, 0], axis=1) / np.sum(
                weights, axis=1
            )
            self.rain = np.sum(weights * nrain[indices][:, :, 0], axis=1) / np.sum(
                weights, axis=1
            )

            if not self.back:
                self.erodep = np.sum(
                    weights * nerodep[indices][:, :, 0], axis=1
                ) / np.sum(weights, axis=1)
                self.sedLoad = np.sum(
                    weights * nsedLoad[indices][:, :, 0], axis=1
                ) / np.sum(weights, axis=1)
                self.flowAcc = np.sum(
                    weights * nflowAcc[indices][:, :, 0], axis=1
                ) / np.sum(weights, axis=1)
                if self.lookuplift and step > 0:
                    # self.hdisp = np.sum(
                    #     weights * nhdisp[indices][:, :, 0], axis=1
                    # ) / np.sum(weights, axis=1)
                    self.uplift = np.sum(
                        weights * nuplift[indices][:, :, 0], axis=1
                    ) / np.sum(weights, axis=1)

        if len(onIDs) > 0:
            self.elev[onIDs] = nelev[indices[onIDs, 0]]
            self.rain[onIDs] = nrain[indices[onIDs, 0]]

            if not self.back:
                self.erodep[onIDs] = nerodep[indices[onIDs, 0]]
                self.sedLoad[onIDs] = nsedLoad[indices[onIDs, 0]]
                self.flowAcc[onIDs] = nflowAcc[indices[onIDs, 0]]
                if self.lookuplift and step > 0:
                    # self.hdisp[onIDs] = nhdisp[indices[onIDs, 0]]
                    self.uplift[onIDs] = nuplift[indices[onIDs, 0]]

        if not self.back:
            if self.lookuplift:
                del weights, nelev, nrain, distances, indices, ncoords
                del nerodep, nsedLoad, nflowAcc
                if step > 0:
                    del nuplift  # nhdisp
            else:
                del (
                    weights,
                    nelev,
                    nrain,
                    distances,
                    indices,
                    ncoords,
                    nsedLoad,
                    nflowAcc,
                )
        else:
            del weights, nelev, nrain, distances, indices, ncoords

        gc.collect()
        return

    def exportVTK(self, vtkfile, sl=0.0):

        mdata = np.load(self.npdata + ".npz")

        self.hFill, self.labels = filllabel(sl, self.elev, mdata["n"])

        vis_mesh = meshio.Mesh(
            self.vertices,
            {"triangle": self.cells},
            point_data={
                "elev": self.elev,
                "erodep": self.erodep,
                "rain": self.rain,
                "FA": np.ma.log(self.flowAcc).filled(0),
                "SL": self.sedLoad,
                "fill": self.hFill - self.elev,
                "basin": self.labels,
            },
        )
        meshio.write(vtkfile, vis_mesh)
        print("Writing VTK file {}".format(vtkfile))

        return

    def exportNetCDF(self, ncfile):

        try:
            os.remove(ncfile)
        except OSError:
            pass

        ds = nc.Dataset(ncfile, "w", format="NETCDF4")
        ds.description = "gospl outputs"
        ds.history = "Created " + time.ctime(time.time())

        if self.nbstep is not None:
            dtime = ds.createDimension("time", None)

        dlat = ds.createDimension("latitude", len(self.lat[:, 0]))
        dlon = ds.createDimension("longitude", len(self.lon[0, :]))

        if self.nbstep is not None:
            times = ds.createVariable("time", "f8", ("time",))
            times.units = "days since 0000-01-01"
            times.calendar = "365_day"
            times[:] = date2num(
                [
                    cftime.num2date(
                        i * 365.0, units="days since 0000-01-01", calendar="365_day",
                    )
                    for i in range(self.nbstep)
                ],
                units=times.units,
                calendar=times.calendar,
            )

        lats = ds.createVariable("latitude", "f8", ("latitude",))
        lats.units = "degrees_north"
        lats[:] = self.lat[:, 0]

        lons = ds.createVariable("longitude", "f8", ("longitude",))
        lons.units = "degrees_east"
        lons[:] = self.lon[0, :]

        if self.nbstep is not None:
            elev = ds.createVariable(
                "elevation", "f8", ("time", "latitude", "longitude"), zlib=True
            )
            elev.units = "metres"
            elev[:, :, :] = self.datafelev
        else:
            elev = ds.createVariable(
                "elevation", "f8", ("latitude", "longitude"), zlib=True
            )
            elev.units = "metres"
            elev[:, :] = self.datafelev

        if self.nbstep is not None:
            erodep = ds.createVariable(
                "erodep", "f8", ("time", "latitude", "longitude"), zlib=True
            )
            erodep.units = "metres"
            erodep[:, :, :] = self.datafEroDep
        else:
            erodep = ds.createVariable(
                "erodep", "f8", ("latitude", "longitude"), zlib=True
            )
            erodep.units = "metres"
            erodep[:, :] = self.datafEroDep

        if self.nbstep is not None:
            rain = ds.createVariable(
                "precipitation", "f8", ("time", "latitude", "longitude"), zlib=True
            )
            rain.units = "m/yr"
            rain[:, :, :] = self.datafRain
        else:
            rain = ds.createVariable(
                "precipitation", "f8", ("latitude", "longitude"), zlib=True
            )
            rain.units = "m/yr"
            rain[:, :] = self.datafRain

        if self.nbstep is not None:
            fla = ds.createVariable(
                "drainageArea", "f8", ("time", "latitude", "longitude"), zlib=True
            )
            fla.units = "m2"
            fla[:, :, :] = self.datafA
        else:
            fla = ds.createVariable(
                "drainageArea", "f8", ("latitude", "longitude"), zlib=True
            )
            fla.units = "m2"
            fla[:, :] = self.datafA

        if self.nbstep is not None:
            fsl = ds.createVariable(
                "sedimentLoad", "f8", ("time", "latitude", "longitude"), zlib=True
            )
            fsl.units = "m3/yr"
            fsl[:, :, :] = self.datafSed
        else:
            fsl = ds.createVariable(
                "sedimentLoad", "f8", ("latitude", "longitude"), zlib=True
            )
            fsl.units = "m3/yr"
            fsl[:, :] = self.datafSed

        if self.lookuplift:

            # if self.nbstep is not None:
            #     fh = ds.createVariable(
            #         "hdisp", "f4", ("time", "latitude", "longitude"), zlib=True
            #     )
            #     fh.units = "m/yr"
            #     fh[:, :, :] = self.datafhdisp
            # else:
            #     fh = ds.createVariable(
            #         "hdisp", "f4", ("latitude", "longitude"), zlib=True
            #     )
            #     fh.units = "m/yr"
            #     fh[:, :] = self.datafhdisp

            if self.nbstep is not None:
                fu = ds.createVariable(
                    "uplift", "f4", ("time", "latitude", "longitude"), zlib=True
                )
                fu.units = "m/yr"
                fu[:, :, :] = self.datafUp
            else:
                fu = ds.createVariable(
                    "uplift", "f4", ("latitude", "longitude"), zlib=True
                )
                fu.units = "m/yr"
                fu[:, :] = self.datafUp

        if self.nbstep is not None:
            fl = ds.createVariable(
                "basinID", "i4", ("time", "latitude", "longitude"), zlib=True
            )
            fl.units = "int"
            fl[:, :, :] = self.datafBasin
        else:
            fl = ds.createVariable(
                "basinID", "i4", ("latitude", "longitude"), zlib=True
            )
            fl.units = "int"
            fl[:, :] = self.datafBasin

        ds.close()

        del ds

        return

    def _readElevationData(self, step):

        self._getCoordinates(step)

        return

    def buildLonLatMesh(self, res=0.1, nghb=3, box=None):

        if self.nx is None:
            if box is None:
                self.nx = int(360.0 / res) + 1
                self.ny = int(180.0 / res) + 1
                self.lon = np.linspace(-180.0, 180.0, self.nx)
                self.lat = np.linspace(-90.0, 90.0, self.ny)
            else:
                self.nx = int((box[2] - box[0]) / res) + 1
                self.ny = int((box[3] - box[1]) / res) + 1
                self.lon = np.linspace(box[0], box[2], self.nx)
                self.lat = np.linspace(box[1], box[3], self.ny)

            self.lon, self.lat = np.meshgrid(self.lon, self.lat)
            self.xyi = np.dstack([self.lon.flatten(), self.lat.flatten()])[0]

        distances, indices = self.tree.query(self.xyi, k=nghb)
        onIDs = np.where(distances[:, 0] == 0)[0]
        distances[onIDs, :] = 0.001
        weights = 1.0 / distances ** 2
        denum = 1.0 / np.sum(weights, axis=1)
        denum[onIDs] = 0.0

        zi = np.sum(weights * self.elev[indices], axis=1) * denum
        raini = np.sum(weights * self.rain[indices], axis=1) * denum
        if not self.back:
            erodepi = np.sum(weights * self.erodep[indices], axis=1) * denum
            sedLoadi = np.sum(weights * self.sedLoad[indices], axis=1) * denum
            if self.lookuplift and self.uplift is not None:
                uplifti = np.sum(weights * self.uplift[indices], axis=1) * denum
                # hdispi = np.sum(weights * self.hdisp[indices], axis=1) * denum

        if len(onIDs) > 0:
            zi[onIDs] = self.elev[indices[onIDs, 0]]
            raini[onIDs] = self.rain[indices[onIDs, 0]]
            if not self.back:
                erodepi[onIDs] = self.erodep[indices[onIDs, 0]]
                sedLoadi[onIDs] = self.sedLoad[indices[onIDs, 0]]
                if self.lookuplift and self.uplift is not None:
                    uplifti[onIDs] = self.uplift[indices[onIDs, 0]]
                    # hdispi[onIDs] = self.hdisp[indices[onIDs, 0]]

        raini = np.reshape(raini, (self.ny, self.nx))
        z = np.reshape(zi, (self.ny, self.nx))

        if not self.back:
            th = np.reshape(erodepi, (self.ny, self.nx))
            sl = np.reshape(sedLoadi, (self.ny, self.nx))
            if self.lookuplift and self.uplift is not None:
                vdisp = np.reshape(uplifti, (self.ny, self.nx))
                # hdisp = np.reshape(hdispi, (self.ny, self.nx))

            elev = z.copy()
            elev[elev < 0.0] = -9999

            class metadata:
                no_data = -9999
                projection = "+init=epsg:4326"
                geotransform = (
                    res,
                    0.0,
                    self.lon.min() - 0.5 * res,
                    0.0,
                    -res,
                    self.lat.max() + 0.5 * res,
                )

            dem = rd.rdarray(elev, meta_obj=metadata, no_data=-9999)
            rd.FillDepressions(dem, epsilon=True, in_place=True)
            M, slp = self.flow_direction_d8(res, res, dem, fill_pits=False)
            FlAc = self.flowAccum(M, (self.ny, self.nx))
            DB, outlets = self.drainageBasins(M, (self.ny, self.nx))
            DB[z < 0] = -1
            FlAc[z < 0] = 0.0001
            FlAc = np.ma.log(FlAc)
            FlAc.filled(0)
            del M, slp, elev, dem

        if self.step == 0 and self.nbstep is not None:
            self.datafelev = np.zeros((self.nbstep, self.ny, self.nx))
            if not self.back:
                self.datafA = np.zeros((self.nbstep, self.ny, self.nx))
                self.datafRain = np.zeros((self.nbstep, self.ny, self.nx))
                self.datafSL = np.zeros((self.nbstep, self.ny, self.nx))
                self.datafSed = np.zeros((self.nbstep, self.ny, self.nx))
                if self.lookuplift:
                    self.datafUp = np.zeros((self.nbstep, self.ny, self.nx))
                    # self.datafhdisp = np.zeros((self.nbstep, self.ny, self.nx))
                self.datafEroDep = np.zeros((self.nbstep, self.ny, self.nx))
                self.datafBasin = np.zeros((self.nbstep, self.ny, self.nx), dtype=int)

        if self.nbstep is not None:
            self.datafelev[self.step, :, :] = z
            self.datafRain[self.step, :, :] = raini

            if not self.back:
                self.datafEroDep[self.step, :, :] = th
                self.datafSed[self.step, :, :] = sl
                self.datafA[self.step, :, :] = FlAc
                self.datafBasin[self.step, :, :] = DB

                if self.lookuplift and self.uplift is not None:
                    self.datafUp[self.step, :, :] = vdisp
                    # self.datafhdisp[self.step, :, :] = hdisp
        else:

            self.datafelev = z
            self.datafRain = raini

            if not self.back:
                self.datafEroDep = th
                self.datafSed = sl
                self.datafA = FlAc
                self.datafBasin = DB

                if self.lookuplift and self.uplift is not None:
                    self.datafUp = vdisp
                    # self.datafhdisp = hdisp

        if not self.back:
            if self.lookuplift:
                del (
                    weights,
                    denum,
                    onIDs,
                    zi,
                    raini,
                    erodepi,
                    sedLoadi,
                    # uplifti,
                    # hdispi,
                    z,
                    th,
                    FlAc,
                    DB,
                    # vdisp,
                    # hdisp,
                )
            else:
                del (weights, denum, onIDs, raini, zi, z, th, FlAc, DB, sedLoadi)
        else:
            del (weights, denum, onIDs, raini, zi, z)
        gc.collect()

        return
