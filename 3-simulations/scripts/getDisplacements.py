import os
import glob
import h5py
import xarray as xr
import meshio
import numpy as np
from scipy import spatial
from scipy import ndimage
import netCDF4 as nc


class getDisplacements:
    def __init__(self, npdata=None, output=None, step=None):

        self.outputDir = output
        self.nbCPUs = len(glob.glob1(self.outputDir + "/h5/", "topology.p*"))
        if self.nbCPUs == 0:
            self.nbCPUs = 1

        # Load mesh structure
        self.npdata = npdata
        mesh_struct = np.load(str(self.npdata))
        self.vertices = mesh_struct["v"]
        self.cells = mesh_struct["c"]
        self.ngbID = mesh_struct["n"]

        self.readElevationData(step)

        return

    def readElevationData(self, step):

        for k in range(self.nbCPUs):
            df = h5py.File("%s/h5/topology.p%s.h5" % (self.outputDir, k), "r")
            coords = np.array((df["/coords"]))

            df2 = h5py.File("%s/h5/gospl.%s.p%s.h5" % (self.outputDir, step, k), "r")
            elev = np.array((df2["/elev"]))

            if k == 0:
                x, y, z = np.hsplit(coords, 3)
                nelev = elev
            else:
                x = np.append(x, coords[:, 0])
                y = np.append(y, coords[:, 1])
                z = np.append(z, coords[:, 2])
                nelev = np.append(nelev, elev)

            df.close()
            df2.close()

        self.nbPts = len(x)
        ncoords = np.zeros((self.nbPts, 3))
        ncoords[:, 0] = x.ravel()
        ncoords[:, 1] = y.ravel()
        ncoords[:, 2] = z.ravel()

        tree = spatial.cKDTree(ncoords, leafsize=10)
        distances, indices = tree.query(self.vertices, k=3)

        # Inverse weighting distance...
        weights = 1.0 / distances ** 2
        onIDs = np.where(distances[:, 0] == 0)[0]
        if nelev[indices].ndim == 2:
            self.elev = np.sum(weights * nelev[indices][:, :], axis=1) / np.sum(
                weights, axis=1
            )
        else:
            self.elev = np.sum(weights * nelev[indices][:, :, 0], axis=1) / np.sum(
                weights, axis=1
            )
        if len(onIDs) > 0:
            self.elev[onIDs] = nelev[indices[onIDs, 0]]

        self._xyz2lonlat()
        self.tree = spatial.cKDTree(self.lonlat, leafsize=10)

        return

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
        # id1 = np.where(self.lonlat[:, 0] < 0)[0]
        # id2 = np.where(self.lonlat[:, 0] >= 0)[0]
        # self.lonlat[id1, 0] += 180.0
        # self.lonlat[id2, 0] -= 180.0
        self.lonlat[:, 0] += 180.0
        self.lonlat[:, 1] += 90.0

        return

    def getPaleoTopo(self, paleoDems=None):

        # Open it with xarray
        data = xr.open_dataset(paleoDems)
        lon_name = "longitude"
        data["_longitude_adjusted"] = xr.where(
            data[lon_name] < 0, data[lon_name] + 360, data[lon_name]
        )
        data = (
            data.swap_dims({lon_name: "_longitude_adjusted"})
            .sel(**{"_longitude_adjusted": sorted(data._longitude_adjusted)})
            .drop(lon_name)
        )
        data = data.rename({"_longitude_adjusted": lon_name})
        paleoData = data.sortby(data.latitude)

        tmp = paleoData.z.values.T
        # Map mesh coordinates on this dataset
        lon1 = tmp.shape[0] * (self.lonlat[:, 0]) / 360.0
        lat1 = tmp.shape[1] * (self.lonlat[:, 1]) / 180.0
        dataxyz = np.stack((lon1, lat1))
        self.next_elev = ndimage.map_coordinates(
            tmp, dataxyz, order=2, mode="nearest"
        ).astype(np.float64)

        self.disp = self.next_elev - self.elev

        return

    def dispMap(self, res=0.1, nghb=3):

        self.nx = int(360.0 / res) + 1
        self.ny = int(180.0 / res) + 1
        self.ilon = np.linspace(0.0, 360.0, self.nx)
        self.ilat = np.linspace(0.0, 180.0, self.ny)
        self.ilon, self.ilat = np.meshgrid(self.ilon, self.ilat)
        self.xyi = np.dstack([self.ilon.flatten(), self.ilat.flatten()])[0]

        self.dists, self.ids = self.tree.query(self.xyi, k=nghb)
        self.oIDs = np.where(self.dists[:, 0] == 0)[0]
        self.dists[self.oIDs, :] = 0.001
        self.wghts = 1.0 / self.dists ** 2
        self.denum = 1.0 / np.sum(self.wghts, axis=1)
        self.denum[self.oIDs] = 0.0

        zi = np.sum(self.wghts * self.elev[self.ids], axis=1) * self.denum
        nzi = np.sum(self.wghts * self.next_elev[self.ids], axis=1) * self.denum
        dispi = np.sum(self.wghts * self.disp[self.ids], axis=1) * self.denum

        if len(self.oIDs) > 0:
            zi[self.oIDs] = self.elev[self.ids[self.oIDs, 0]]
            nzi[self.oIDs] = self.next_elev[self.ids[self.oIDs, 0]]
            dispi[self.oIDs] = self.disp[self.ids[self.oIDs, 0]]

        self.ielev = np.reshape(zi, (self.ny, self.nx))
        self.inelev = np.reshape(nzi, (self.ny, self.nx))
        self.idisp = np.reshape(dispi, (self.ny, self.nx))

        id1 = np.where(self.ilon[0, :] - 180.0 <= 0)[0]
        id2 = np.where(self.ilon[0, :] - 180.0 >= 0)[0]

        self.idelev = self.ielev.copy()
        self.idelev[:, id1] = self.ielev[:, id2]
        self.idelev[:, id2] = self.ielev[:, id1]

        self.idnelev = self.inelev.copy()
        self.idnelev[:, id1] = self.inelev[:, id2]
        self.idnelev[:, id2] = self.inelev[:, id1]

        self.indisp = self.idisp.copy()
        self.indisp[:, id1] = self.idisp[:, id2]
        self.indisp[:, id2] = self.idisp[:, id1]

        return

    def getSmoothDisp(self, ndisp):

        itree = spatial.cKDTree(self.xyi, leafsize=10)

        id1 = np.where(self.ilon[0, :] - 180.0 >= 0)[0]
        id2 = np.where(self.ilon[0, :] - 180.0 <= 0)[0]

        sdisp = ndisp.copy()
        sdisp[:, id1] = ndisp[:, id2]
        sdisp[:, id2] = ndisp[:, id1]

        fdisp = sdisp.flatten()

        dists, ids = itree.query(self.lonlat, k=3)
        oIDs = np.where(dists[:, 0] == 0)[0]
        dists[oIDs, :] = 0.001
        wghts = 1.0 / dists ** 2
        denum = 1.0 / np.sum(wghts, axis=1)
        denum[oIDs] = 0.0
        self.sdisps = np.sum(wghts * fdisp[ids], axis=1) * denum
        if len(oIDs) > 0:
            self.sdisps[oIDs] = fdisp[ids[oIDs, 0]]

        return

    def exportVTK(self, vtkfile, sl=0.0):

        vis_mesh = meshio.Mesh(
            self.vertices,
            {"triangle": self.cells},
            point_data={
                "elev": self.elev,
                "nelev": self.next_elev,
                "disp": self.disp,
                "ndisp": self.sdisps,
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
        ds.description = "tectonic forcing"

        dlat = ds.createDimension("latitude", len(self.ilat[:, 0]))
        dlon = ds.createDimension("longitude", len(self.ilon[0, :]))

        lats = ds.createVariable("latitude", "f8", ("latitude",))
        lats.units = "degrees_north"
        lats[:] = self.ilat[:, 0] - 90

        lons = ds.createVariable("longitude", "f8", ("longitude",))
        lons.units = "degrees_east"
        lons[:] = self.ilon[0, :] - 180

        elev = ds.createVariable("Z", "f8", ("latitude", "longitude"), zlib=True)
        elev.units = "metres"
        elev[:, :] = self.idelev

        nelev = ds.createVariable("nZ", "f8", ("latitude", "longitude"), zlib=True)
        nelev.units = "metres"
        nelev[:, :] = self.idnelev

        disp = ds.createVariable("disp", "f8", ("latitude", "longitude"), zlib=True)
        disp.units = "metres"
        disp[:, :] = self.indisp

        ds.close()

        return
