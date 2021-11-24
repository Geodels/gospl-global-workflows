import gc
import sys
import glob
import h5py
import numba as nb
import numpy as np
from scipy import spatial
import ruamel.yaml as yaml
from pyevtk.hl import gridToVTK
from scipy.ndimage import gaussian_filter


@nb.jit(nopython=True)
def getVals(k, idlat1, idlat2, idlon1, idlon2, lon, lat, zz, zi, thu, th, phiSi, topz):

    shape = (idlon2 - idlon1, idlat2 - idlat1)
    lx = np.empty(shape)
    ly = np.empty(shape)
    lz = np.empty(shape)
    le = np.empty(shape)
    lh = np.empty(shape)
    lps = np.empty(shape)
    lt = np.empty(shape)

    for j in range(shape[1]):
        for i in range(shape[0]):
            lx[i, j] = lon[i + idlon1]
            ly[i, j] = lat[j + idlat1]
            if topz is None:
                lz[i, j] = zz[j, i]
            else:
                lz[i, j] = topz[i, j] - thu[j, i]
            le[i, j] = zi[j, i]
            lh[i, j] = th[j, i]
            lps[i, j] = phiSi[j, i]
            lt[i, j] = k

    return lx, ly, lz, le, lh, lps, lt


@nb.jit(nopython=True)
def getVals2(
    k,
    idlat1,
    idlat2,
    idlon1,
    idlon2,
    lon,
    lat,
    zz,
    zi,
    thu,
    th,
    phiSi,
    finei,
    phiFi,
    weathi,
    phiWi,
    topz,
):

    shape = (idlon2 - idlon1, idlat2 - idlat1)
    lx = np.empty(shape)
    ly = np.empty(shape)
    lz = np.empty(shape)
    le = np.empty(shape)
    lh = np.empty(shape)
    lps = np.empty(shape)
    lt = np.empty(shape)
    lf = np.empty(shape)
    lw = np.empty(shape)
    lpf = np.empty(shape)
    lpw = np.empty(shape)

    for j in range(shape[1]):
        for i in range(shape[0]):
            lx[i, j] = lon[i + idlon1]
            ly[i, j] = lat[j + idlat1]
            if topz is None:
                lz[i, j] = zz[j, i]
            else:
                lz[i, j] = topz[i, j] - thu[j, i]
            le[i, j] = zi[j, i]
            lh[i, j] = th[j, i]
            lf[i, j] = finei[j, i]
            lw[i, j] = weathi[j, i]
            lps[i, j] = phiSi[j, i]
            lpf[i, j] = phiFi[j, i]
            lpw[i, j] = phiWi[j, i]
            lt[i, j] = k

    return lx, ly, lz, le, lh, lf, lw, lps, lpf, lpw, lt


class stratal:
    def __init__(self, path=None, filename=None, layer=None, model="spherical"):

        self.path = path
        if path is not None:
            filename = self.path + filename

        # Check input file exists
        try:
            with open(filename) as finput:
                pass
        except IOError:
            print("Unable to open file: ", filename)
            raise IOError("The input file is not found...")

        # Open YAML file
        with open(filename, "r") as finput:
            self.input = yaml.load(finput, Loader=yaml.Loader)

        self.radius = 6378137.0
        self._inputParser()

        self.nbCPUs = len(glob.glob1(self.outputDir + "/h5/", "topology.p*"))

        if layer is not None:
            self.layNb = layer
        else:
            self.layNb = len(glob.glob1(self.outputDir + "/h5/", "stratal.*.p0.h5"))

        print("Created sedimentary layers:", self.layNb)

        self.nbfile = len(glob.glob1(self.outputDir + "/h5/", "stratal.*.p0.h5"))

        self.utm = False
        if model != "spherical":
            self.utm = True

        return

    def _inputParser(self):

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
            self.tEnd = timeDict["end"]
        except KeyError:
            print("Key 'end' is required and is missing in the 'time' declaration!")
            raise KeyError("Simulation end time needs to be declared.")

        try:
            self.strat = timeDict["strat"]
        except KeyError:
            print(
                "Key 'strat' is required to build the stratigraphy in the input file!"
            )
            raise KeyError("Simulation stratal time needs to be declared.")

        domainDict = self.input["domain"]
        try:
            strataFile = domainDict["npstrata"]
            self.strataFile = strataFile + ".npz"
            with open(self.strataFile) as strataFile:
                strataFile.close()
        except KeyError:
            self.strataFile = None

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

        return

    def _xyz2lonlat(self):

        r = np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

        lons = np.arctan2(self.y, self.x)
        lats = np.arcsin(self.z / r)

        # Convert spherical mesh longitudes and latitudes to degrees
        self.lonlat = np.empty((self.nbPts, 2))
        self.lonlat[:, 0] = np.mod(np.degrees(lons) + 180.0, 360.0) - 180.0
        self.lonlat[:, 1] = np.mod(np.degrees(lats) + 90, 180.0) - 90.0
        id1 = np.where(self.lonlat[:, 0] < 0)[0]
        id2 = np.where(self.lonlat[:, 0] >= 0)[0]
        self.lonlat[id1, 0] += 180.0
        self.lonlat[id2, 0] -= 180.0
        self.tree = spatial.cKDTree(self.lonlat, leafsize=10)

        return

    def lonlat2xyz(self, lon, lat, radius=6378137.0):

        rlon = np.radians(lon)
        rlat = np.radians(lat)

        coords = np.zeros((3))
        coords[0] = np.cos(rlat) * np.cos(rlon) * radius
        coords[1] = np.cos(rlat) * np.sin(rlon) * radius
        coords[2] = np.sin(rlat) * radius

        return coords

    def _getCoordinates(self):

        for k in range(self.nbCPUs):
            df = h5py.File("%s/h5/topology.p%s.h5" % (self.outputDir, k), "r")
            coords = np.array((df["/coords"]))
            if k == 0:
                self.x, self.y, self.z = np.hsplit(coords, 3)
            else:
                self.x = np.append(self.x, coords[:, 0])
                self.y = np.append(self.y, coords[:, 1])
                self.z = np.append(self.z, coords[:, 2])
            df.close()

        self.nbPts = len(self.x)
        if not self.utm:
            self._xyz2lonlat()
        else:
            self.lonlat = np.empty((len(self.x), 2))
            self.lonlat[:, 0] = self.x
            self.lonlat[:, 1] = self.y
            self.tree = spatial.cKDTree(self.lonlat, leafsize=10)

        gc.collect()

        return

    def getData(self, nbCPUs, outputDir, nbfile, strataFile):

        for k in range(nbCPUs):
            sf = h5py.File("%s/h5/stratal.%s.p%s.h5" % (outputDir, nbfile, k), "r")
            if k == 0:
                elev = np.array(sf["/stratZ"])
                th = np.array(sf["/stratH"])
                phiS = np.array(sf["/phiS"])
                if strataFile is not None:
                    fine = np.array(sf["/stratF"])
                    phiF = np.array(sf["/phiF"])
                    weathered = np.array(sf["/stratW"])
                    phiW = np.array(sf["/phiW"])
            else:
                elev = np.append(elev, sf["/stratZ"], axis=0)
                th = np.append(th, sf["/stratH"], axis=0)
                phiS = np.append(phiS, sf["/phiS"], axis=0)
                if strataFile is not None:
                    fine = np.append(fine, sf["/stratF"], axis=0)
                    phiF = np.append(phiF, sf["/phiF"], axis=0)
                    weathered = np.append(weathered, sf["/stratW"], axis=0)
                    phiW = np.append(phiW, sf["/phiW"], axis=0)
            sf.close()

        if strataFile is None:
            fine = None
            phiF = None
            weathered = None
            phiW = None

        return elev, th, phiS, fine, phiF, weathered, phiW

    def readStratalData(self):

        self._getCoordinates()

        (
            self.elev,
            self.th,
            self.phiS,
            self.fine,
            self.phiF,
            self.weath,
            self.phiW,
        ) = self.getData(self.nbCPUs, self.outputDir, self.layNb, self.strataFile)

        self.curLay = self.th.shape[1]
        print("Number of sedimentary layers:", self.curLay)

        return

    def _test_progress(self, job_title, progress):

        length = 20
        block = int(round(length * progress))
        msg = "\r{0}: [{1}] {2}%".format(
            job_title, "#" * block + "-" * (length - block), round(progress * 100, 2)
        )
        if progress >= 1:
            msg += " DONE\r\n"
        sys.stdout.write(msg)
        sys.stdout.flush()

    def buildLonLatMesh(self, res=0.1, nghb=3):

        self.lon = np.arange(-180.0, 180.0 + res, res)
        self.nx = len(self.lon)
        self.lat = np.arange(-90.0, 90.0 + res, res)
        self.ny = len(self.lat)

        self.lon, self.lat = np.meshgrid(self.lon, self.lat)
        xyi = np.dstack([self.lon.flatten(), self.lat.flatten()])[0]
        self.zi = np.empty((self.curLay, self.ny, self.nx))
        self.thi = np.empty((self.curLay, self.ny, self.nx))
        self.phiSi = np.empty((self.curLay, self.ny, self.nx))

        if self.strataFile is not None:
            self.phiFi = np.empty((self.curLay, self.ny, self.nx))
            self.finei = np.empty((self.curLay, self.ny, self.nx))
            self.phiWi = np.empty((self.curLay, self.ny, self.nx))
            self.weathi = np.empty((self.curLay, self.ny, self.nx))

        distances, indices = self.tree.query(xyi, k=nghb)
        dist = distances.copy()
        dist[dist == 0] = 1.0e-6
        weights = 1.0 / dist ** 2
        denum = 1.0 / np.sum(weights, axis=1)
        onIDs = np.where(distances[:, 0] == 0)[0]

        print("Start building regular stratigraphic arrays")

        for k in range(self.curLay):

            zz = self.elev[:, k]
            th = self.th[:, k]
            phiS = self.phiS[:, k]
            self._test_progress("Percentage of arrays built ", (k + 1) / self.curLay)

            zi = np.sum(weights * zz[indices], axis=1) * denum
            thi = np.sum(weights * th[indices], axis=1) * denum
            phiSi = np.sum(weights * phiS[indices], axis=1) * denum

            if self.strataFile is not None:
                fine = self.fine[:, k]
                phiF = self.phiF[:, k]
                weath = self.weath[:, k]
                phiW = self.phiW[:, k]
                finei = np.sum(weights * fine[indices], axis=1) * denum
                phiFi = np.sum(weights * phiF[indices], axis=1) * denum
                weathi = np.sum(weights * weath[indices], axis=1) * denum
                phiWi = np.sum(weights * phiW[indices], axis=1) * denum

            if len(onIDs) > 0:
                zi[onIDs] = zz[indices[onIDs, 0]]
                thi[onIDs] = th[indices[onIDs, 0]]
                phiSi[onIDs] = phiS[indices[onIDs, 0]]

                if self.strataFile is not None:
                    finei[onIDs] = fine[indices[onIDs, 0]]
                    phiFi[onIDs] = phiF[indices[onIDs, 0]]
                    weathi[onIDs] = weath[indices[onIDs, 0]]
                    phiWi[onIDs] = phiW[indices[onIDs, 0]]

            self.zi[k, :, :] = np.reshape(zi, (self.ny, self.nx))
            self.thi[k, :, :] = np.reshape(thi, (self.ny, self.nx))
            self.phiSi[k, :, :] = np.reshape(phiSi, (self.ny, self.nx))
            if self.strataFile is not None:
                self.finei[k, :, :] = np.reshape(finei, (self.ny, self.nx))
                self.phiFi[k, :, :] = np.reshape(phiFi, (self.ny, self.nx))
                self.weathi[k, :, :] = np.reshape(weathi, (self.ny, self.nx))
                self.phiWi[k, :, :] = np.reshape(phiWi, (self.ny, self.nx))

        return

    def buildUTMmesh(self, res=5000.0, nghb=3):

        xo = self.x.min()
        xm = self.x.max()
        yo = self.y.min()
        ym = self.y.max()

        self.lon = np.arange(xo, xm + res, res)
        self.lat = np.arange(yo, ym + res, res)
        self.nx = len(self.lon)
        self.ny = len(self.lat)

        self.lon, self.lat = np.meshgrid(self.lon, self.lat)
        xyi = np.dstack([self.lon.flatten(), self.lat.flatten()])[0]
        self.zi = np.empty((self.curLay, self.ny, self.nx))
        self.thi = np.empty((self.curLay, self.ny, self.nx))
        self.phiSi = np.empty((self.curLay, self.ny, self.nx))
        if self.strataFile is not None:
            self.finei = np.empty((self.curLay, self.ny, self.nx))
            self.weathi = np.empty((self.curLay, self.ny, self.nx))
            self.phiFi = np.empty((self.curLay, self.ny, self.nx))
            self.phiWi = np.empty((self.curLay, self.ny, self.nx))

        distances, indices = self.tree.query(xyi, k=nghb)
        weights = 1.0 / (distances + 0.000001) ** 2
        denum = 1.0 / np.sum(weights, axis=1)
        onIDs = np.where(distances[:, 0] < 0.000001)[0]

        print("Start building regular stratigraphic arrays")

        for k in range(self.curLay):

            zz = self.elev[:, k]
            th = self.th[:, k]
            phiS = self.phiS[:, k]
            if self.strataFile is not None:
                wth = self.wth[:, k]
                fine = self.fine[:, k]
                phiF = self.phiF[:, k]
                phiW = self.phiW[:, k]
            self._test_progress("Percentage of arrays built ", (k + 1) / self.curLay)
            zi = np.sum(weights * zz[indices], axis=1) * denum
            thi = np.sum(weights * th[indices], axis=1) * denum
            phiSi = np.sum(weights * phiS[indices], axis=1) * denum
            if self.strataFile is not None:
                wthi = np.sum(weights * wth[indices], axis=1) * denum
                finei = np.sum(weights * fine[indices], axis=1) * denum
                phiFi = np.sum(weights * phiF[indices], axis=1) * denum
                phiWi = np.sum(weights * phiW[indices], axis=1) * denum

            if self.strataFile is not None:
                zi[onIDs] = zz[indices[onIDs, 0]]
                thi[onIDs] = th[indices[onIDs, 0]]
                phiSi[onIDs] = phiS[indices[onIDs, 0]]
                if self.strataFile is not None:
                    finei[onIDs] = fine[indices[onIDs, 0]]
                    wthi[onIDs] = wth[indices[onIDs, 0]]
                    phiWi[onIDs] = phiW[indices[onIDs, 0]]

            self.zi[k, :, :] = np.reshape(zi, (self.ny, self.nx))
            self.thi[k, :, :] = np.reshape(thi, (self.ny, self.nx))
            self.phiSi[k, :, :] = np.reshape(phiSi, (self.ny, self.nx))

            if self.strataFile is not None:
                self.wthi[k, :, :] = np.reshape(wthi, (self.ny, self.nx))
                self.finei[k, :, :] = np.reshape(finei, (self.ny, self.nx))
                self.phiFi[k, :, :] = np.reshape(phiFi, (self.ny, self.nx))
                self.phiWi[k, :, :] = np.reshape(phiWi, (self.ny, self.nx))

        return

    def writeMesh(self, vtkfile="mesh", lons=None, lats=None, sigma=0.0):
        """
        Create a vtk unstructured grid based on current time step stratal parameters.
        """

        if not self.utm:
            lon = np.linspace(-180.0, 180.0, self.nx)
            lat = np.linspace(-90.0, 90.0, self.ny)

            if lons is None:
                idlon1 = 0
                idlon2 = len(lon)
            else:
                idlon1 = np.where(self.lon[0, :] >= lons[0])[0].min()
                idlon2 = np.where(self.lon[0, :] >= lons[1])[0].min()

            if lats is None:
                idlat1 = 0
                idlat2 = len(lat)
            else:
                idlat1 = np.where(self.lat[:, 0] >= lats[0])[0].min()
                idlat2 = np.where(self.lat[:, 0] >= lats[1])[0].min()

            shape = (idlon2 - idlon1, idlat2 - idlat1, self.curLay)
        else:
            xo = self.x.min()
            xm = self.x.max()
            yo = self.y.min()
            ym = self.y.max()
            lon = np.linspace(xo, xm, self.nx)
            lat = np.linspace(yo, ym, self.ny)
            lons = [0, self.nx]
            lats = [0, self.ny]
            idlat1 = 0
            idlat2 = len(lat)
            idlon1 = 0
            idlon2 = len(lon)
            shape = (self.nx, self.ny, self.curLay)

        x = np.empty(shape)
        y = np.empty(shape)
        z = np.empty(shape)
        e = np.empty(shape)
        h = np.empty(shape)
        t = np.empty(shape)
        ps = np.empty(shape)

        if self.strataFile is not None:
            w = np.empty(shape)
            pw = np.empty(shape)
            f = np.empty(shape)
            pf = np.empty(shape)

        zz = self.zi[-1, idlat1:idlat2, idlon1:idlon2]
        zz = gaussian_filter(zz, sigma)

        for k in range(self.curLay - 1, -1, -1):
            if sigma > 0:
                th = gaussian_filter(self.thi[k, idlat1:idlat2, idlon1:idlon2], sigma)
            th[th < 0] = 0.0
            if k < self.curLay - 1:
                if sigma > 0:
                    thu = gaussian_filter(
                        self.thi[k + 1, idlat1:idlat2, idlon1:idlon2], sigma
                    )
                thu[thu < 0] = 0.0
            else:
                thu = None
            zi = self.zi[k, idlat1:idlat2, idlon1:idlon2]
            phiSi = self.phiSi[k, idlat1:idlat2, idlon1:idlon2]
            if self.strataFile is not None:
                finei = self.finei[k, idlat1:idlat2, idlon1:idlon2]
                phiFi = self.phiFi[k, idlat1:idlat2, idlon1:idlon2]
                weathi = self.weathi[k, idlat1:idlat2, idlon1:idlon2]
                phiWi = self.phiWi[k, idlat1:idlat2, idlon1:idlon2]

            if k == self.curLay - 1:
                topz = None
            else:
                topz = z[:, :, k + 1]

            if self.strataFile is not None:
                (
                    x[:, :, k],
                    y[:, :, k],
                    z[:, :, k],
                    e[:, :, k],
                    h[:, :, k],
                    f[:, :, k],
                    w[:, :, k],
                    ps[:, :, k],
                    pf[:, :, k],
                    pw[:, :, k],
                    t[:, :, k],
                ) = (
                    getVals2(
                        k,
                        idlat1,
                        idlat2,
                        idlon1,
                        idlon2,
                        lon,
                        lat,
                        zz,
                        zi,
                        thu,
                        th,
                        phiSi,
                        finei,
                        phiFi,
                        weathi,
                        phiWi,
                        topz,
                    ),
                )
            else:
                print(k, idlat1, idlat2, idlon1, idlon2, zi.shape, zz.shape)
                (
                    x[:, :, k],
                    y[:, :, k],
                    z[:, :, k],
                    e[:, :, k],
                    h[:, :, k],
                    ps[:, :, k],
                    t[:, :, k],
                ) = (
                    getVals(
                        k,
                        idlat1,
                        idlat2,
                        idlon1,
                        idlon2,
                        lon,
                        lat,
                        zz,
                        zi,
                        thu,
                        th,
                        phiSi,
                        topz,
                    ),
                )

        if self.strataFile is not None:
            gridToVTK(
                vtkfile,
                x,
                y,
                z,
                pointData={
                    "dep elev": e,
                    "th": h,
                    "layID": t,
                    "percweath": w,
                    "percfine": f,
                    "phiC": ps,
                    "phiF": pf,
                    "phiW": pw,
                },
            )
        else:
            gridToVTK(
                vtkfile,
                x,
                y,
                z,
                pointData={"dep elev": e, "th": h, "layID": t, "phiC": ps,},
            )

        return

        lon = np.linspace(-180.0, 180.0, self.nx)
        lat = np.linspace(-90.0, 90.0, self.ny)

        if lons is None:
            lons = [lon[0], lon[1]]
        else:
            lons[0] = np.where(self.lon[0, :] >= lons[0])[0].min()
            lons[1] = np.where(self.lon[0, :] >= lons[1])[0].min()
        if lats is None:
            lats = [lat[0], lat[1]]
        else:
            lats[0] = np.where(self.lat[:, 0] >= lats[0])[0].min()
            lats[1] = np.where(self.lat[:, 0] >= lats[1])[0].min()

        x = np.zeros((lons[1] - lons[0], lats[1] - lats[0], self.curLay))
        y = np.zeros((lons[1] - lons[0], lats[1] - lats[0], self.curLay))
        z = np.zeros((lons[1] - lons[0], lats[1] - lats[0], self.curLay))
        e = np.zeros((lons[1] - lons[0], lats[1] - lats[0], self.curLay))
        h = np.zeros((lons[1] - lons[0], lats[1] - lats[0], self.curLay))
        t = np.zeros((lons[1] - lons[0], lats[1] - lats[0], self.curLay))
        ps = np.zeros((lons[1] - lons[0], lats[1] - lats[0], self.curLay))

        if self.strataFile is not None:
            w = np.zeros((lons[1] - lons[0], lats[1] - lats[0], self.curLay))
            pw = np.zeros((lons[1] - lons[0], lats[1] - lats[0], self.curLay))
            f = np.zeros((lons[1] - lons[0], lats[1] - lats[0], self.curLay))
            pf = np.zeros((lons[1] - lons[0], lats[1] - lats[0], self.curLay))

        zz = self.zi[-1, lats[0] : lats[1], lons[0] : lons[1]]
        if sigma > 0:
            zz = gaussian_filter(zz, sigma)

        # res = 360.0 / self.nx
        # hscale = 110000.0 * res
        # xmax = -1.0e12
        # ymax = -1.0e12

        for k in range(self.curLay - 1, -1, -1):
            if sigma > 0:
                th = gaussian_filter(self.thi[k, :, :], sigma)
            th[th < 0] = 0.0
            if k < self.curLay - 1:
                if sigma > 0:
                    thu = gaussian_filter(self.thi[k + 1, :, :], sigma)
                thu[thu < 0] = 0.0
            for j in range(lats[1] - lats[0]):
                for i in range(lons[1] - lons[0]):
                    x[i, j, k] = lon[i + lons[0]] - lon[lons[0]]  # * hscale
                    y[i, j, k] = lat[j + lats[0]] - lat[lats[0]]  # * hscale
                    # xmax = max(xmax, x[i, j, k])
                    # ymax = max(ymax, y[i, j, k])
                    if k == self.curLay - 1:
                        z[i, j, k] = zz[j, i]
                    else:
                        z[i, j, k] = z[i, j, k + 1] - thu[j + lats[0], i + lons[0]]
                    e[i, j, k] = self.zi[k, j + lats[0], i + lons[0]]
                    h[i, j, k] = th[j + lats[0], i + lons[0]]
                    ps[i, j, k] = self.phiSi[k, j + lats[0], i + lons[0]]
                    t[i, j, k] = k
                    if self.strataFile is not None:
                        f[i, j, k] = self.finei[k, j + lats[0], i + lons[0]]
                        pf[i, j, k] = self.phiFi[k, j + lats[0], i + lons[0]]
                        w[i, j, k] = self.weathi[k, j + lats[0], i + lons[0]]
                        pw[i, j, k] = self.phiWi[k, j + lats[0], i + lons[0]]

        if self.strataFile is not None:
            gridToVTK(
                vtkfile,
                x,
                y,
                z,
                pointData={
                    "dep elev": e,
                    "th": h,
                    "layID": t,
                    "percweath": w,
                    "percfine": f,
                    "phiC": ps,
                    "phiF": pf,
                    "phiW": pw,
                },
            )
        else:
            gridToVTK(
                vtkfile,
                x,
                y,
                z,
                pointData={"dep elev": e, "th": h, "layID": t, "phiC": ps,},
            )

        return  # [xmax, ymax]
