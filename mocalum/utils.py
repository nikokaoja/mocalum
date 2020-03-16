
import numpy as np
import xarray as xr
from math import degrees, atan2
from .persistance import data

def spher2cart(azimuth, elevation, radius):
    """Converts spherical coordinates to Cartesian coordinates

    Parameters
    ----------
    azimuth : numpy
        Array containing azimuth angles
    elevation : numpy
        Array containing elevation angles
    radius : numpy
        Array containing radius angles

    Returns
    -------
    x : numpy
        Array containing x coordinate values.
    y : numpy
        Array containing y coordinate values.
    z : numpy
        Array containing z coordinate values.

    Raises
    ------
    TypeError
        If dimensions of inputs are not the same
    """

    azimuth = 90 - azimuth # converts to 'Cartesian' angle
    azimuth = np.radians(azimuth) # converts to radians
    elevation = np.radians(elevation)

    try:
        x = radius * np.cos(azimuth) * np.cos(elevation)
        y = radius * np.sin(azimuth) * np.cos(elevation)
        z = radius * np.sin(elevation)
        return x, y, z
    except:
        raise TypeError('Dimensions of inputs are not the same!')



def project2los(u,v,w, azimuth, elevation, ignore_elevation = True):
    """Projects wind vector to line-of-sight

    Parameters
    ----------
    u : numpy
        u component of the wind
    v : numpy
        v component of the wind
    w : numpy
        upward velocity of the wind
    azimuth : numpy
        azimuth angle of the beam
    elevation : numpy
        elevation angle of the beam
    ignore_elevation : bool, optional
        To assume horizontal beam or not, by default True

    Returns
    -------
    los : numpy
        Projected wind vector to LOS, i.e. radial/los wind speed
    Raises
    ------
    TypeError
        If dimensions of inputs are not the same
    """

    # handles both single values as well arrays
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)

    try:
        if ignore_elevation:
            los = u * np.sin(azimuth) + v * np.cos(azimuth)
        else:
            los = u * np.sin(azimuth) * np.cos(elevation) + \
                v * np.cos(azimuth) * np.cos(elevation) + \
                w * np.sin(elevation)

        return los
    except:
        raise TypeError('Dimensions of inputs are not the same!')



def ivap_rc(los, azimuth, ax = 0):
    """Calculates u and v components by using IVAP algo on set of LOS speed
    measurements acquired using PPI scans.

    Parameters
    ----------
    los : numpy
        Array containing LOS speed measurements
    azimuth : numpy
        Array containing azimuth angles corresponding to LOS speed measurements
    ax : int
        Along which axes should calculation take place, by default 0

    Returns
    -------
    u : numpy
        Array of reconstracted u component of the wind
    v : numpy
        Array of reconstracted v component of the wind
    wind_speed: numpy
        Array of reconstracted horizontal wind speed
    """
    azimuth = np.radians(azimuth)

    A = ((np.sin(azimuth))**2).sum(axis=ax)
    B = ((np.cos(azimuth))**2).sum(axis=ax)
    C = (np.sin(azimuth)*np.cos(azimuth)).sum(axis=ax)
    D = A*B - C**2
    E = (los*np.cos(azimuth)).sum(axis=ax)
    F = (los*np.sin(azimuth)).sum(axis=ax)

    v = (E*A - F*C) / D
    u = (F*B - E*C) / D
    wind_speed = np.sqrt(u**2 + v**2)

    return u, v, wind_speed

def dd_rc(los, azimuth):
    """Calculates u and v components by using IVAP algo on set of LOS speed
    measurements acquired using PPI scans.

    Parameters
    ----------
    los : numpy
        Array containing LOS speed measurements
    azimuth : numpy
        Array containing azimuth angles corresponding to LOS speed measurements

    Returns
    -------
    u : numpy
        Array of reconstracted u component of the wind
    v : numpy
        Array of reconstracted v component of the wind
    wind_speed: numpy
        Array of reconstracted horizontal wind speed
    """

    azimuth = np.radians(azimuth)

    R_ws = np.array([[np.sin(azimuth[0]), np.cos(azimuth[0])],
                     [np.sin(azimuth[1]), np.cos(azimuth[1])]])

    u, v = np.dot(np.linalg.inv(R_ws),los)
    wind_speed = np.sqrt(u**2 + v**2)

    wind_direction = (90 - degrees(atan2(-v, -u))) % 360

    return u, v, wind_speed, wind_direction



def td_rc(los, azimuth, elevation):
    """Calculates u and v components by using IVAP algo on set of LOS speed
    measurements acquired using PPI scans.

    Parameters
    ----------
    los : numpy
        Array containing LOS speed measurements
    azimuth : numpy
        Array containing azimuth angles corresponding to LOS speed measurements

    Returns
    -------
    u : numpy
        Array of reconstracted u component of the wind
    v : numpy
        Array of reconstracted v component of the wind
    wind_speed: numpy
        Array of reconstracted horizontal wind speed
    """
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)
    R_ws = np.array([[np.cos(elevation[0]) * np.sin(azimuth[0]), np.cos(elevation[0]) * np.cos(azimuth[0]), np.sin(elevation[0])],
                     [np.cos(elevation[1]) * np.sin(azimuth[1]), np.cos(elevation[1]) * np.cos(azimuth[1]), np.sin(elevation[1])],
                     [np.cos(elevation[2]) * np.sin(azimuth[2]), np.cos(elevation[2]) * np.cos(azimuth[2]), np.sin(elevation[2])]])

    u, v, w = np.dot(np.linalg.inv(R_ws),los)
    wind_speed = np.sqrt(u**2 + v**2)
    wind_direction = (90 - degrees(atan2(-v, -u))) % 360

    return u,v,w, wind_speed, wind_direction


def move2time(displacement, Amax, Vmax):
    """
    Calculates minimum move time to perform predefined angular motions.
    """

    max_acc_move = (Vmax**2) / Amax

    if displacement > max_acc_move:
        return displacement/Vmax + Vmax/Amax
    else:
        return 2*np.sqrt(displacement/Amax)

def get_ivap_probing(sector_size, azimuth_mid, angular_res,
                     elevation, distance, no_scans = 1,
                     scan_speed=1, max_speed=50, max_acc=100):

    az = np.arange(azimuth_mid-sector_size/2,
                   azimuth_mid+sector_size/2 + angular_res,
                   angular_res, dtype=float)

    no_los = len(az)

    az = np.tile(az, no_scans)
    dis = np.full(len(az), distance, dtype=float)
    el = np.full(len(az), elevation, dtype=float)
    xyz= np.full(len(az), np.nan, dtype=float)
    unc = np.full(len(az), 0, dtype=float)

    # setting up time dim
    time = np.arange(0, sector_size*scan_speed + angular_res*scan_speed,
                     angular_res*scan_speed)
    time = np.tile(time, no_scans)
    sweep_time = move2time(sector_size, max_acc, max_speed)
    scan_time = sector_size*scan_speed
    to_add = np.repeat(scan_time + sweep_time, no_scans*no_los)
    multip = np.repeat(np.arange(0,no_scans), no_los)
    time = time + multip*to_add

    data._cr8_los_ds(az, el, dis,)

    # # needs to add scan_id as dimension
    # ds = xr.Dataset({'az': (['time'], az),
    #                  'el': (['time'], el),
    #                  'dis': (['time'], dis),
    #                  'x': (['time'], xyz),
    #                  'y': (['time'], xyz),
    #                  'z': (['time'], xyz),
    #                  'unc_az': (['time'], unc),
    #                  'unc_el': (['time'], unc),
    #                  'unc_dis': (['time'], unc),
    #                  'unc_los': (['time'], unc),
    #                  'u': (['time'], unc),
    #                  'v': (['time'], unc),
    #                  'w': (['time'], unc),
    #                  'v_rad': (['time'], unc),
    #                  'sector_size':(sector_size),
    #                  'no_scans':(no_scans),
    #                  'no_los':(no_los),
    #                  'scan_time':(scan_time),
    #                  'sweep_back_time':(sweep_time),
    #                  },

    #                 coords={'time': time})

    # return ds


def add_xyz(ds, lidar_pos):
    x,y,z = spher2cart(ds.az +ds.unc_az,
                       ds.el +ds.unc_el,
                       ds.dis+ds.unc_dis)
    ds.x.values = x.values + lidar_pos[0]
    ds.y.values = y.values + lidar_pos[1]
    ds.z.values = z.values + lidar_pos[2]

    return ds

def get_plaw_uvw(height, ref_height=100, wind_speed=10,
                 w=0, wind_dir=180, shear_exponent=0.2):
    u = - wind_speed * np.sin(np.radians(wind_dir))
    v = - wind_speed * np.cos(np.radians(wind_dir))
    gain = (height / ref_height)**shear_exponent
    u_new = gain * u
    v_new = gain * v
    w_new = np.full(len(u_new), w)

    return u_new, v_new, w_new


def _fill_in(empty_array, values):
    full_array = np.copy(empty_array)
    for i, value in enumerate(values):
        full_array[i, :, :] = value
    return full_array


def gen_field_pl(ds, ref_height=100, wind_speed=10, w=0, wind_dir=180, shear_exponent=0.2):
    x_coords = np.arange(ds.x.min(), ds.x.max() + 1, 10)
    y_coords = np.arange(ds.y.min(), ds.y.max() + 1, 10)
    z_coords = np.arange(ds.z.min(), ds.z.max() + 1, 1)

    base_array = np.empty((len(z_coords), len(y_coords),len(x_coords)), dtype=float)
    u, v, w = get_uvw_pl(z_coords, ref_height, wind_speed, w, wind_dir, shear_exponent)

    u_array = _fill_in(base_array, u)
    v_array = _fill_in(base_array, v)
    w_array = _fill_in(base_array, w)

    ds_wind = xr.Dataset({'u': (['z', 'y', 'x'], u_array),
                          'v': (['z', 'y', 'x'], v_array),
                          'w': (['z', 'y', 'x'], w_array)},
                         coords={'x': x_coords, 'y': y_coords, 'z':z_coords})

    return ds_wind


def inject_wind_data(ds, ds_wind):

    ds.u.values = ds_wind.u.interp(x=ds.x.values.mean(),
                                   y=ds.y.values.mean(),
                                   z = ds.z.values)
    ds.v.values = ds_wind.v.interp(x=ds.x.values.mean(),
                                   y=ds.y.values.mean(),
                                   z = ds.z.values)
    ds.w.values = ds_wind.w.interp(x=ds.x.values.mean(),
                                   y=ds.y.values.mean(),
                                   z = ds.z.values)
    ds.v_rad.values = project2los(ds.u.values,
                                  ds.v.values,
                                  ds.w.values,
                                  ds.az.values + ds.unc_az.values,
                                  ds.el.values + ds.unc_el.values,
                                  ignore_elevation = True)

    ds.v_rad.values = ds.v_rad.values + ds.unc_los.values

    return ds

