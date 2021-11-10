import os
import glob
import h5py
import meshio
import time
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy import spatial
import ruamel.yaml as yaml
from scipy.interpolate import interp1d

from gospl._fortran import filllabel


class mapOutputs:
    def __init__(self, path=None, filename=None, step=None, uplift=True):

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

        self.res = None
        self.step = step
        self.nbPts = 0
        self.radius = 6378137.0
        self.lookuplift = uplift
        self._inputParser()
        self.nx = None
        self.uplift = None

        self.nbCPUs = len(glob.glob1(self.outputDir + "/h5/", "topology.p*"))

        self.getData(step)

        self.datafA = None
        self.datafSed = None
        self.datafRain = None
        self.datafelev = None
        self.datafUp = None
        self.datafhdisp = None
        self.datafEroDep = None
        self.datafBasin = None

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

        xs = np.array(self.vertices[:, 0])
        ys = np.array(self.vertices[:, 1])
        zs = np.array(self.vertices[:, 2] / r)

        lons = np.arctan2(ys, xs)
        lats = np.arcsin(zs)

        # Convert spherical mesh longitudes and latitudes to degrees
        self.lonlat = np.empty((len(self.vertices[:, 0]), 2))
        self.lonlat[:, 0] = np.mod(np.degrees(lons) + 180.0, 360.0) - 180.0
        self.lonlat[:, 1] = np.mod(np.degrees(lats) + 90, 180.0) - 90.0
        id1 = np.where(self.lonlat[:, 0] < 0)[0]
        id2 = np.where(self.lonlat[:, 0] >= 0)[0]
        self.lonlat[id1, 0] += 180.0
        self.lonlat[id2, 0] -= 180.0
        self.tree = spatial.cKDTree(self.lonlat, leafsize=10)

        return

    def getData(self, step):

        if self.nbCPUs == 0:
            self.nbCPUs = 1

        for k in range(self.nbCPUs):

            if self.nbPts == 0:
                df = h5py.File("%s/h5/topology.p%s.h5" % (self.outputDir, k), "r")
                coords = np.array((df["/coords"]))

            df2 = h5py.File("%s/h5/gospl.%s.p%s.h5" % (self.outputDir, step, k), "r")
            elev = np.array((df2["/elev"]))
            rain = np.array((df2["/rain"]))
            erodep = np.array((df2["/erodep"]))
            sedLoad = np.array((df2["/sedLoad"]))
            fillAcc = np.array((df2["/fillFA"]))
            flowAcc = np.array((df2["/flowAcc"]))
            if self.lookuplift and step > 0:
                uplift = np.array((df2["/uplift"]))

            if self.seacurve:
                sealevel = self.seafunction(self.time[step])
                elev -= sealevel
            else:
                elev -= self.sealevel

            if k == 0:
                if self.nbPts == 0:
                    x, y, z = np.hsplit(coords, 3)
                nelev = elev
                nrain = rain
                nerodep = erodep
                nsedLoad = sedLoad
                nflowAcc = flowAcc
                nfillAcc = fillAcc
                if self.lookuplift and step > 0:
                    nuplift = uplift
            else:
                if self.nbPts == 0:
                    x = np.append(x, coords[:, 0])
                    y = np.append(y, coords[:, 1])
                    z = np.append(z, coords[:, 2])
                nelev = np.append(nelev, elev)
                nrain = np.append(nrain, rain)
                nerodep = np.append(nerodep, erodep)
                nsedLoad = np.append(nsedLoad, sedLoad)
                nfillAcc = np.append(nfillAcc, fillAcc)
                nflowAcc = np.append(nflowAcc, flowAcc)
                if self.lookuplift and step > 0:
                    nuplift = np.append(nuplift, uplift)
            if self.nbPts == 0:
                df.close()
            df2.close()

        if self.nbPts == 0:
            self.nbPts = len(x)
            ncoords = np.zeros((self.nbPts, 3))
            ncoords[:, 0] = x.ravel()
            ncoords[:, 1] = y.ravel()
            ncoords[:, 2] = z.ravel()
            # Load mesh structure
            mesh_struct = np.load(str(self.npdata) + ".npz")
            self.vertices = mesh_struct["v"]
            self.cells = mesh_struct["c"]
            self.ngbID = mesh_struct["n"]
            self._xyz2lonlat()
            tree = spatial.cKDTree(ncoords, leafsize=10)
            self.distances, self.indices = tree.query(self.vertices, k=3)
            # Inverse weighting distance...
            self.weights = 1.0 / self.distances ** 2
            self.onIDs = np.where(self.distances[:, 0] == 0)[0]
            self.sumwght = np.sum(self.weights, axis=1)

        if nelev[self.indices].ndim == 2:
            self.elev = (
                np.sum(self.weights * nelev[self.indices][:, :], axis=1) / self.sumwght
            )
            self.rain = (
                np.sum(self.weights * nrain[self.indices][:, :], axis=1) / self.sumwght
            )
            self.erodep = (
                np.sum(self.weights * nerodep[self.indices][:, :], axis=1)
                / self.sumwght
            )
            self.sedLoad = (
                np.sum(self.weights * nsedLoad[self.indices][:, :], axis=1)
                / self.sumwght
            )
            self.fillAcc = (
                np.sum(self.weights * nfillAcc[self.indices][:, :], axis=1)
                / self.sumwght
            )
            self.flowAcc = (
                np.sum(self.weights * nflowAcc[self.indices][:, :], axis=1)
                / self.sumwght
            )
            if self.lookuplift and step > 0:
                self.uplift = (
                    np.sum(self.weights * nuplift[self.indices][:, :], axis=1)
                    / self.sumwght
                )

        else:
            self.elev = (
                np.sum(self.weights * nelev[self.indices][:, :, 0], axis=1)
                / self.sumwght
            )
            self.rain = (
                np.sum(self.weights * nrain[self.indices][:, :, 0], axis=1)
                / self.sumwght
            )
            self.erodep = (
                np.sum(self.weights * nerodep[self.indices][:, :, 0], axis=1)
                / self.sumwght
            )
            self.sedLoad = (
                np.sum(self.weights * nsedLoad[self.indices][:, :, 0], axis=1)
                / self.sumwght
            )
            self.flowAcc = (
                np.sum(self.weights * nflowAcc[self.indices][:, :, 0], axis=1)
                / self.sumwght
            )
            self.fillAcc = (
                np.sum(self.weights * nfillAcc[self.indices][:, :, 0], axis=1)
                / self.sumwght
            )
            if self.lookuplift and step > 0:
                self.uplift = (
                    np.sum(self.weights * nuplift[self.indices][:, :, 0], axis=1)
                    / self.sumwght
                )

        if len(self.onIDs) > 0:
            self.elev[self.onIDs] = nelev[self.indices[self.onIDs, 0]]
            self.rain[self.onIDs] = nrain[self.indices[self.onIDs, 0]]
            self.erodep[self.onIDs] = nerodep[self.indices[self.onIDs, 0]]
            self.sedLoad[self.onIDs] = nsedLoad[self.indices[self.onIDs, 0]]
            self.flowAcc[self.onIDs] = nflowAcc[self.indices[self.onIDs, 0]]
            self.fillAcc[self.onIDs] = nfillAcc[self.indices[self.onIDs, 0]]
            if self.lookuplift and step > 0:
                self.uplift[self.onIDs] = nuplift[self.indices[self.onIDs, 0]]

        mdata = np.load(self.npdata + ".npz")
        # nelev = self.elev.copy()
        # ids = np.where(self.flowAcc)[0]
        # nelev[ids] = -1000.0
        self.hFill, self.labels = filllabel(0.0, self.elev, mdata["n"])

        return

    def exportVTK(self, vtkfile):

        if self.lookuplift:
            vis_mesh = meshio.Mesh(
                self.vertices,
                {"triangle": self.cells},
                point_data={
                    "elev": self.elev,
                    "erodep": self.erodep,
                    "rain": self.rain,
                    "FA": np.ma.log(self.flowAcc).filled(0),
                    "fillFA": np.ma.log(self.fillAcc).filled(0),
                    "SL": self.sedLoad,
                    "fill": self.hFill - self.elev,
                    "basin": self.labels,
                    "vtec": self.uplift,
                },
            )
        else:
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

    def buildLonLatMesh(self, res=0.1, nghb=3, box=None):

        if self.res is not None:
            if self.res != res:
                self.nx = None

        if self.nx is None:
            self.res = res
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

            self.dists, self.ids = self.tree.query(self.xyi, k=nghb)
            self.oIDs = np.where(self.dists[:, 0] == 0)[0]
            self.dists[self.oIDs, :] = 0.001
            self.wghts = 1.0 / self.dists ** 2
            self.denum = 1.0 / np.sum(self.wghts, axis=1)
            self.denum[self.oIDs] = 0.0

        zi = np.sum(self.wghts * self.elev[self.ids], axis=1) * self.denum
        fai = np.sum(self.wghts * self.flowAcc[self.ids], axis=1) * self.denum
        ffai = np.sum(self.wghts * self.fillAcc[self.ids], axis=1) * self.denum
        raini = np.sum(self.wghts * self.rain[self.ids], axis=1) * self.denum
        erodepi = np.sum(self.wghts * self.erodep[self.ids], axis=1) * self.denum
        sedLoadi = np.sum(self.wghts * self.sedLoad[self.ids], axis=1) * self.denum
        if self.lookuplift and self.uplift is not None:
            uplifti = np.sum(self.wghts * self.uplift[self.ids], axis=1) * self.denum
        lbli = self.labels[self.ids[:, 0]]

        if len(self.oIDs) > 0:
            zi[self.oIDs] = self.elev[self.ids[self.oIDs, 0]]
            raini[self.oIDs] = self.rain[self.ids[self.oIDs, 0]]
            fai[self.oIDs] = self.flowAcc[self.ids[self.oIDs, 0]]
            ffai[self.oIDs] = self.fillAcc[self.ids[self.oIDs, 0]]
            erodepi[self.oIDs] = self.erodep[self.ids[self.oIDs, 0]]
            sedLoadi[self.oIDs] = self.sedLoad[self.ids[self.oIDs, 0]]
            if self.lookuplift and self.uplift is not None:
                uplifti[self.oIDs] = self.uplift[self.ids[self.oIDs, 0]]

        raini = np.reshape(raini, (self.ny, self.nx))
        z = np.reshape(zi, (self.ny, self.nx))
        th = np.reshape(erodepi, (self.ny, self.nx))
        sl = np.reshape(sedLoadi, (self.ny, self.nx))
        fa = np.reshape(fai, (self.ny, self.nx))
        ffa = np.reshape(ffai, (self.ny, self.nx))
        lbl = np.reshape(lbli, (self.ny, self.nx))
        if self.lookuplift and self.uplift is not None:
            vdisp = np.reshape(uplifti, (self.ny, self.nx))

        if self.datafA is None:
            self.datafelev = np.zeros((self.ny, self.nx))
            self.datafA = np.zeros((self.ny, self.nx))
            self.dataffA = np.zeros((self.ny, self.nx))
            self.datafRain = np.zeros((self.ny, self.nx))
            self.datafSL = np.zeros((self.ny, self.nx))
            self.datafSed = np.zeros((self.ny, self.nx))
            if self.lookuplift:
                self.datafUp = np.zeros((self.ny, self.nx))
            self.datafEroDep = np.zeros((self.ny, self.nx))
            self.datafBasin = np.zeros((self.ny, self.nx), dtype=int)

        self.datafelev[:, :] = z
        self.datafRain[:, :] = raini
        self.datafEroDep[:, :] = th
        self.datafSed[:, :] = sl
        self.datafA[:, :] = fa
        self.dataffA[:, :] = ffa
        self.datafBasin[:, :] = lbl

        if self.lookuplift and self.uplift is not None:
            self.datafUp[:, :] = vdisp

        return

    def exportNetCDF(self, ncfile):

        try:
            os.remove(ncfile)
        except OSError:
            pass

        ds = nc.Dataset(ncfile, "w", format="NETCDF4")
        ds.description = "gospl outputs"
        ds.history = "Created " + time.ctime(time.time())

        dlat = ds.createDimension("latitude", len(self.lat[:, 0]))
        dlon = ds.createDimension("longitude", len(self.lon[0, :]))

        lats = ds.createVariable("latitude", "f8", ("latitude",))
        lats.units = "degrees_north"
        lats[:] = self.lat[:, 0]

        lons = ds.createVariable("longitude", "f8", ("longitude",))
        lons.units = "degrees_east"
        lons[:] = self.lon[0, :]

        elev = ds.createVariable(
            "elevation", "f8", ("latitude", "longitude"), zlib=True
        )
        elev.units = "metres"
        elev[:, :] = self.datafelev

        erodep = ds.createVariable("erodep", "f8", ("latitude", "longitude"), zlib=True)
        erodep.units = "metres"
        erodep[:, :] = self.datafEroDep

        rain = ds.createVariable(
            "precipitation", "f8", ("latitude", "longitude"), zlib=True
        )
        rain.units = "m/yr"
        rain[:, :] = self.datafRain

        ffla = ds.createVariable(
            "fillDischarge", "f8", ("latitude", "longitude"), zlib=True
        )
        ffla.units = "m3/yr"
        ffla[:, :] = self.dataffA

        fla = ds.createVariable(
            "flowDischarge", "f8", ("latitude", "longitude"), zlib=True
        )
        fla.units = "m3/yr"
        fla[:, :] = self.datafA

        fsl = ds.createVariable(
            "sedimentLoad", "f8", ("latitude", "longitude"), zlib=True
        )
        fsl.units = "m3/yr"
        fsl[:, :] = self.datafSed

        if self.lookuplift:
            fu = ds.createVariable("uplift", "f4", ("latitude", "longitude"), zlib=True)
            fu.units = "m/yr"
            fu[:, :] = self.datafUp

        fl = ds.createVariable("basinID", "i4", ("latitude", "longitude"), zlib=True)
        fl.units = "int"
        fl[:, :] = self.datafBasin

        ds.close()

        del ds

        return
