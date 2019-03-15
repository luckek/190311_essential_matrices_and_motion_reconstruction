import numpy as np
from scipy.optimize import least_squares


class OutOfSensorBoundsError(Exception):
    pass


class InverseCamera(object):

    def __init__(self):

        self.x_pt = None  # Recovered X point

    def estimate_points(self, cam_list, u_gcp, X0):

        self.x_pt = X0.copy()

        # Note: u_gcp should be of the form [[u1, v1],
        #                                    [u2, v2]]
        def residuals(x_pt, cam_list, u_gcp):

            self.x_pt = x_pt.copy()
            x_pt = x_pt.reshape((3, 1))

            # Note: we make res_uv s.t. it is of the same form as u_gcp
            res_uv = np.zeros(shape=(len(cam_list), 2))

            for i, cam in enumerate(cam_list):

                # Note: cam.eve_to_camera gives [[u], [v]]
                res_uv[i] = cam.ene_to_camera(x_pt).ravel()

            # R = [[ f1(X)u - u1, f1(X)v - v1, f2(X)u - u2, f2(X)v - v2, ..., fi(X)u - ui, fi(X)v - vi]
            return res_uv.ravel() - u_gcp.ravel()

        res = least_squares(residuals, self.x_pt, args=(cam_list, u_gcp))
        print(res.cost, '\n')
        return res.x


def read_gcp(fname):
    u = []
    v = []
    east = []
    north = []
    ele = []
    with open(fname) as fd:
        for i, line in enumerate(fd):
            # if i < 2 or line.startswith("#"):
            #     skip header and commented lines
                # continue
            vals = line.split(",")[1:-1]
            u.append(int(float(vals[0].strip())))
            v.append(int(float(vals[1].strip())))
            east.append(float(vals[2].strip()))
            north.append(float(vals[3].strip()))
            ele.append(float(vals[4].strip()))
    uv = np.vstack((u, v))
    ene = np.vstack((east, north, ele))
    return uv, ene


def read_gcp_nh(gcp_fname):

    uv_ene = np.loadtxt(gcp_fname, delimiter=',')
    uv, ene = uv_ene[:, 0:2], uv_ene[:, 2:]

    return uv, ene
