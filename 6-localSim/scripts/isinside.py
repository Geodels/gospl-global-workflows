import math
import numba
import pygplates
import numpy as np
import pandas as pd
from scipy import spatial
from numba import jit, njit
from pyproj import Transformer

import pyvista as pv
import triangle as triangle

from shapely.geometry import LineString
import matplotlib.path as mpltPath
from scipy.ndimage import gaussian_filter

# Check if a point is within the coastline polygon
@jit(nopython=True)
def is_inside_sm(polygon, point):
    length = len(polygon) - 1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii < length:
        dy = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy * dy2 <= 0.0 and (
            point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]
        ):

            # non-horizontal line
            if dy < 0 or dy2 < 0:
                F = dy * (polygon[jj][0] - polygon[ii][0]) / (dy - dy2) + polygon[ii][0]

                if (
                    point[0] > F
                ):  # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F:  # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2 == 0 and (
                point[0] == polygon[jj][0]
                or (
                    dy == 0
                    and (point[0] - polygon[ii][0]) * (point[0] - polygon[jj][0]) <= 0
                )
            ):
                return 2

        ii = jj
        jj += 1

    # print 'intersections=", intersections
    return intersections & 1


@njit(parallel=True)
def is_inside_sm_parallel(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean)
    for i in numba.prange(ln):
        D[i] = is_inside_sm(polygon, points[i])
    return D


def getRivers(file, src_epsg, dst_epsg):

    features = pygplates.FeatureCollection(file)
    polypts = []
    for feature in features:
        polypts.append(feature.get_geometry().to_lat_lon_array())

    nb = 0
    path = []
    for k in range(len(polypts)):
        path.append(mpltPath.Path(polypts[k]).vertices)
        nb += len(path[-1])

    # Extract the ID, latitude and longitude of the river's points
    data = np.zeros((nb, 3))

    stp = 0
    for k in range(len(polypts)):
        lnb = len(path[k])
        data[stp : stp + lnb, 0] = k
        data[stp : stp + lnb, 1] = path[k][:, 0]
        data[stp : stp + lnb, 2] = path[k][:, 1]
        stp += lnb

    # Change from lon/lat to utm
    transformer = Transformer.from_crs(src_epsg, dst_epsg)
    riverXY = np.asarray([transformer.transform(x, y) for x, y in data[:, 1:]])

    riverdf = pd.DataFrame(
        {
            "x": riverXY[:, 0],
            "y": riverXY[:, 1],
            "lon": data[:, 2],
            "lat": data[:, 1],
            "id": data[:, 0],
        }
    )

    return riverdf


def getCoast(file, points, elev, src_epsg, dst_epsg, coastres, slp):

    features = pygplates.FeatureCollection(file)
    shape = elev.shape

    # Read polygon values
    polypts = []
    for feature in features:
        polypts.append(feature.get_geometry().to_lat_lon_array())

    nb = 0
    path = []
    for k in range(len(polypts)):
        path.append(mpltPath.Path(polypts[k]).vertices)
        nb += len(path[-1])

    data = np.zeros((nb + 1, 3))
    stp = 0
    for k in range(len(polypts)):
        lnb = len(path[k])
        data[stp : stp + lnb, 0] = k
        data[stp : stp + lnb, 1] = path[k][:, 0]
        data[stp : stp + lnb, 2] = path[k][:, 1]
        stp += lnb

    data[-1, :] = data[0, :]

    # Change from lon/lat to utm
    transformer = Transformer.from_crs(src_epsg, dst_epsg)
    coastXY = np.asarray([transformer.transform(x, y) for x, y in data[:, 1:]])

    coastdf = pd.DataFrame({"x": coastXY[:, 0], "y": coastXY[:, 1]})

    # Create more points along the coastline
    xy = []
    for p in range(len(coastdf.x) - 1):
        ls = LineString(
            [(coastdf.x[p], coastdf.y[p]), (coastdf.x[p + 1], coastdf.y[p + 1])]
        )
        for f in range(0, int(math.ceil(ls.length)) + 1, int(coastres)):
            p = ls.interpolate(f).coords[0]
            xy.append((p[0], p[1]))
    coast = np.array(xy)

    inside = is_inside_sm_parallel(points, coast)

    # Build a tree with the coordinates of the coastline
    coastTree = spatial.cKDTree(coast, leafsize=10)

    # Find the distance of the 2D mesh to the coastline
    bsl = np.where(inside + 0 < 1)[0]  # points below coastline
    distanceMar = np.zeros(len(points))
    dist, ids = coastTree.query(points[bsl, :2], k=1)
    distanceMar[bsl] = dist

    # Find the distance of the 2D mesh to the coastline
    asl = np.where(inside + 0 >= 1)[0]  # points above coastline
    distanceLand = np.zeros(len(points))
    dist, ids = coastTree.query(points[asl, :2], k=1)
    distanceLand[asl] = dist

    minhLand = slp * distanceLand
    minhMarin = -slp * distanceMar

    # Given an elevation depending on the distance to the coast
    nelev = elev.flatten().copy()
    aboveZ = nelev[asl]
    minH = minhLand[asl]
    id = np.where(aboveZ < minH)[0]
    aboveZ[id] = minH[id]
    nelev[asl] = aboveZ

    belowZ = nelev[bsl]
    minH = minhMarin[bsl]
    id = np.where(belowZ > minH)[0]
    belowZ[id] = minH[id]
    nelev[bsl] = belowZ

    return coast, nelev.reshape(shape)


def riverElev(riverdf, trunkID, elev, points, res, slp):

    ntopo = elev.flatten()

    rivID = []
    rivDF = []
    elevRiv = []

    meshtree = spatial.cKDTree(points, leafsize=10)
    trunk = -np.ones(len(riverdf))

    for k in range(len(trunkID)):

        # Combine each trunk to form the main paleo-rivers
        for p in range(len(trunkID[k])):
            ids = riverdf["id"].values == trunkID[k][p]
            trunk[ids] = k

        id = np.where(trunk == k)[0]

        rivID.append(id)
        rivDF.append(
            pd.DataFrame(
                {
                    "lat": riverdf["lat"].values[id],
                    "lon": riverdf["lon"].values[id],
                    "x": riverdf["x"].values[id],
                    "y": riverdf["y"].values[id],
                }
            )
        )

        # Order from mouth to headwaters
        exist = np.ones(len(id), dtype=bool)
        mouthID = rivDF[-1].lat.argmin()
        criv = np.vstack((rivDF[-1].x, rivDF[-1].y)).T
        tree = spatial.cKDTree(criv)

        order = []
        order.append(mouthID)
        for k in range(1, len(rivDF[-1].x)):
            id = order[-1]
            exist[id] = False
            pos = [rivDF[-1].x[id], rivDF[-1].y[id]]
            d, n = tree.query(pos, k=10)
            for p in range(10):
                if exist[n[p]] == True:
                    id = n[p]
                    break
            order.append(id)

        rivorder = np.zeros(len(rivDF[-1].x))
        for k in range(0, len(rivDF[-1].x)):
            rivorder[order[k]] = k

        rivDF[-1]["order"] = rivorder
        rivDF[-1] = rivDF[-1].sort_values(by="order").reset_index(drop=True)

        # Create more points along each trunk (250 m interval)
        xy = []
        for p in range(len(rivDF[-1].x) - 1):
            ls = LineString(
                [
                    (rivDF[-1].x[p], rivDF[-1].y[p]),
                    (rivDF[-1].x[p + 1], rivDF[-1].y[p + 1]),
                ]
            )
            for f in range(0, int(math.ceil(ls.length)) + 1, int(res)):
                p = ls.interpolate(f).coords[0]
                xy.append((p[0], p[1]))
        xx = np.array(xy, "i")
        nx = xx[:, 0]
        ny = xx[:, 1]

        ds = np.zeros(len(nx))
        for k in range(1, len(nx)):
            ds[k] = ((nx[k] - nx[k - 1]) ** 2 + (ny[k] - ny[k - 1]) ** 2) ** 0.5
            ds[k] += ds[k - 1]

        # Get elevation on each of these trunks based on initial topography
        rivcoords = np.zeros((len(nx), 2))
        rivcoords[:, 0] = nx
        rivcoords[:, 1] = ny
        dist, idd = meshtree.query(rivcoords, k=4)
        # Inverse weighting distance...
        weights = np.divide(1.0, dist, out=np.zeros_like(dist), where=dist != 0)
        onIDs = np.where(dist[:, 0] == 0)[0]
        temp = np.sum(weights, axis=1)
        tmp = np.sum(weights * ntopo[idd], axis=1)
        relev = np.divide(tmp, temp, out=np.zeros_like(temp), where=temp != 0)

        if len(onIDs) > 0:
            relev[onIDs] = ntopo[idd[onIDs, 0]]

        # Ensure downstream flow, you can define different slopes
        distx = ds[::-1]
        rivh = relev[::-1]
        newh = np.zeros(len(distx))
        slp = slp
        newh[0] = rivh[0]

        for k in range(1, len(distx)):
            dh = slp * (distx[k - 1] - distx[k])
            if rivh[k] < newh[k - 1] + dh:
                newh[k] = rivh[k]
            else:
                newh[k] = newh[k - 1] + dh

        datafm = pd.DataFrame(
            {
                "x": nx,
                "y": ny,
                "h": relev,
                "d": ds,
                "nh": newh[::-1],
                "o": np.arange(len(nx)),
            }
        )

        datafm = datafm[datafm.h > 0]
        elevRiv.append(datafm.reset_index())

    return elevRiv


def dataGPML(file, key="m"):

    features = pygplates.FeatureCollection(file)
    contourValue = []
    contourLonLat = []

    for feature in features:
        stringC = feature.get_name()
        newstrC = stringC.replace(key, "")
        contourValue.append(int(newstrC))
        latlon = feature.get_geometry().to_lat_lon_array()
        latlon = np.vstack((latlon, latlon[0]))
        contourLonLat.append(np.vstack((latlon[:, 1], latlon[:, 0])).T)

    contourValue = np.asarray(contourValue)

    return contourLonLat, contourValue


def contour2map(coords, lonlat, val, shape, sigma=2):
    # Then we find the values of the accumulated sediment based on the contours lines
    data = np.zeros(shape)
    for k in range(len(lonlat)):
        inside = is_inside_sm_parallel(coords, lonlat[k])
        boolMat = inside.reshape(shape)
        data[boolMat] = val[k]

    # Let's store it in the xarray dataset
    return gaussian_filter((data), sigma=sigma)


def delaunayMesh(bbox, res):

    nx = int(round((bbox[2] - bbox[0]) / res + 1))
    e_x = np.linspace(bbox[0], bbox[2], nx)
    tmp1 = np.zeros(nx)
    tmp2 = np.zeros(nx)
    tmp1.fill(bbox[1])
    tmp2.fill(bbox[3])
    south = np.column_stack((e_x, tmp1))
    north = np.column_stack((e_x, tmp2))

    ny = int(round((bbox[3] - bbox[1]) / res + 1))
    e_y = np.linspace(bbox[1] + res, bbox[3] - res, ny - 2)
    tmp1 = np.zeros(ny - 2)
    tmp2 = np.zeros(ny - 2)
    tmp1.fill(bbox[0])
    tmp2.fill(bbox[2])
    east = np.column_stack((tmp1, e_y))
    west = np.column_stack((tmp2, e_y))

    # Merge edges together
    edges = []
    edges = np.vstack((south, east))
    edges = np.vstack((edges, north))
    edges = np.vstack((edges, west))

    # Triangulate
    tinMesh = triangle.triangulate({"vertices": edges}, "eDqa" + str(res ** 2))
    ptsTIN = tinMesh["vertices"]

    # Get cells
    coords = np.zeros((len(ptsTIN), 3))
    coords[:, :2] = ptsTIN
    cloudPts = pv.PolyData(coords)
    surface = cloudPts.delaunay_2d()
    faces = surface.faces.reshape((-1, 4))[:, 1:4]

    return ptsTIN, faces
