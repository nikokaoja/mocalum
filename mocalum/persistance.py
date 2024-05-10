"""This module contains a class which stores data created in the interaction
with mocalum.
"""
import time
from . import metadata
import numpy as np
from numpy.linalg import inv as inv
import xarray as xr
from tqdm import tqdm
from .utils import sliding_window_slicing, bbox_pts_from_array, bbox_pts_from_cfg

class Data:
    def __init__(self):
        self.probing = {}
        self.los = {} # should be key-value pairs
        self.ffield = None # should be key-value pairs
        self._ffield = None # should be key-value pairs
        self.rc_wind = None # should be key-value pairs
        self.tenMin_wind = None # should be key-value pairs
        self.sonic_wind_tenMin = None
        self.fmodel_cfg = {}
        self.meas_cfg = {}
        self.bbox_meas_pts = {}
        self.bbox_ffield = {}
        self.ffield_bbox_cfg = {} # should be a dict with lidar_id as key!


    def _cr8_fmodel_cfg(self, cfg):
        """
        Adds configuration of flow model to class

        Parameters
        ----------
        cfg : dict
            Dictionary contating key-value paris which parametrize flow model
        """
        self.fmodel_cfg = cfg


    def _cr8_bbox_dict(self, bbox_type, key, CRS,
                       x_coord, y_coord, z_coord,t_coord,
                       x_offset, y_offset, z_offset,t_offset,
                       x_res, y_res, z_res, t_res, **kwargs):
        """
        Creates bounding box dictionary corresponding to measurement points

        Parameters
        ----------
        bbox_type : str
            Indicates type of bounding box, it can be 'lidar' or 'flow-field'
        key : str
            Key which will be used to add bounding box dict to dict of bboxes
        CRS : dict
            Dict describing coordinate reference system
        x_coord : numpy
            An array of x coordinates of measurement points
        y_coord : numpy
            An array of y coordinates of measurement points
        z_coord : numpy
            An array of z coordinates of measurement points
        t_coord : numpy
             An array of time instance at which measurement points are acquired
        x_offset : float
            Offset of x coordinates
        y_offset : float
            Offset of y coordinates
        z_offset : float
            Offset of z coordinates
        t_offset : float
            Time offset
        x_res : int
            Resolution of x coordinates
        y_res : int
            Resolution of y coordinates
        z_res : int
            Resolution of z coordinates
        t_res : float
            Resolution of time
        """

        bbox_cfg = {}

        # info about coordinate system reference (CRS)
        bbox_cfg.update({'CRS':{'x':CRS['x'],
                                'y':CRS['y'],
                                'z':CRS['z'],
                                'rot_matrix':CRS['rot_matrix']}})


        bbox_cfg.update({'x':{'min':np.min(x_coord),
                              'max':np.max(x_coord),
                              'offset':x_offset,
                              'res':x_res}})

        bbox_cfg.update({'y':{'min':np.min(y_coord),
                              'max':np.max(y_coord),
                              'offset':y_offset,
                              'res':y_res}})

        bbox_cfg.update({'z':{'min':np.min(z_coord),
                              'max':np.max(z_coord),
                              'offset':z_offset,
                              'res':z_res}})

        bbox_cfg.update({'t':{'min':np.min(t_coord),
                              'max':np.max(t_coord),
                              'offset':t_offset,
                              'res':t_res}})

        if bbox_type == 'lidar':
            self.bbox_meas_pts.update({key:bbox_cfg})
        else:
            bbox_cfg.update({'linked_lidars':kwargs['linked_lidars']})
            self.bbox_ffield.update({key:bbox_cfg})


    def _cr8_3d_tfield_ds(self, id, turb_df):
        """
        Creates Mocalum 3D flow field xr.DataSet

        Parameters
        ----------
        id : str
            ID of bounding box cfg
        turb_df : pandas
            PyConTurb pandas df containing 3D turbulence (y,z, time)
        """

        _ , y, z, t = self._get_ffield_coords(id)

        turb_np = turb_df.to_numpy().transpose().ravel()
        turb_np = turb_np.reshape(int(len(turb_np)/len(t)), len(t))


        # -1 to aligned properly axis
        R_tb = -self.bbox_ffield[id]['CRS']['rot_matrix']

        # rotate u and v component to be Eastward and Northward wind
        # according to the met conventions
        uv = np.array([turb_np[0::3],turb_np[1::3]]).transpose()
        tmp_shape = uv.shape
        uv = uv.reshape(tmp_shape[0]*tmp_shape[1],2).dot(inv(R_tb)).reshape(tmp_shape)
        uv = uv.transpose()

        u = uv[0].reshape(len(y), len(z) ,len(t)).transpose(1,0,2)
        v = uv[1].reshape(len(y), len(z) ,len(t)).transpose(1,0,2)
        w = turb_np[2::3].reshape(len(y), len(z) ,len(t)).transpose(1,0,2)

        self.ffield = xr.Dataset({'u': (['z', 'y', 'time'], u),
                                  'v': (['z', 'y', 'time'], v),
                                  'w': (['z', 'y', 'time'], w)},
                                 coords={'time': t,
                                         'y': y,
                                         'z': z})


        # Adding metadata
        self.ffield = self._add_metadata(self.ffield, metadata,
                                         'Turbulent flow field dataset')
        self.ffield.attrs['generator'] = 'PyConTurb'

    def _cr8_4d_tfield_ds(self, id):
        """
        Converts 3D turbulence flow field dataset to 4D dataset

        Parameters
        ----------
        id : str
            ID of bounding box cfg
        """

        self._ffield = self.ffield
        R_tb = self.bbox_ffield[id]['CRS']['rot_matrix']
        y = self.ffield.y.values
        z = self.ffield.z.values


        ws = self.fmodel_cfg['wind_speed']
        bbox_pts = bbox_pts_from_cfg(self.bbox_ffield[id])
        t_res = self.bbox_ffield[id]['t']['res']

        x_start_pos = bbox_pts[:,0].min()
        x_len = abs(bbox_pts[:,0].max() - bbox_pts[:,0].min())
        x_res = ws * t_res
        no_items = int(np.ceil(x_len / x_res)) + 1

        u_3d = self.ffield.u.values.transpose()
        v_3d = self.ffield.v.values.transpose()
        w_3d = self.ffield.w.values.transpose()

        u_4d = sliding_window_slicing(u_3d, no_items, item_type=1).transpose(0,3,2,1)
        v_4d = sliding_window_slicing(v_3d, no_items, item_type=1).transpose(0,3,2,1)
        w_4d = sliding_window_slicing(w_3d, no_items, item_type=1).transpose(0,3,2,1)

        t = np.arange(0, u_4d.shape[0]*t_res, t_res)
        if t.shape[0]>u_4d.shape[0]:
            t = np.delete(t, t.shape[0]-1)  # hack for something that shouldn't happen but it does
        x = np.arange(0, no_items*x_res, x_res) + x_start_pos


        ew = np.empty((len(x),len(y),2))



        for i in range(0,len(x)):
            for j in range(0,len(y)):
                ew[i,j] = np.array([x[i],y[j]]).dot(inv(R_tb))


        self.ffield = xr.Dataset({'u': (['time', 'z', 'y', 'x'], u_4d),
                                  'v': (['time', 'z', 'y', 'x'], v_4d),
                                  'w': (['time', 'z', 'y', 'x'], w_4d)},
                                coords={'time': t,
                                        'y': y,
                                        'z': z,
                                        'x': x,
                                         'Easting' : (['x','y'], ew[:,:,0]),
                                         'Northing' : (['x','y'], ew[:,:,1]),
                                         'Height' : (['z'], z)
                                        })

        self.ffield.attrs['generator'] = 'turbulence_box'
        self.ffield = self._add_metadata(self.ffield, metadata,
                                         'Turbulent flow field dataset')

    def _cr8_plfield_ds(self, bbox_id, u, v, w):
        """
        Creates power law flow field 3D dataset

        Parameters
        ----------
        bbox_id : str
            ID of bounding box cfg
        u : numpy
            ND array of shape (len(z), len(y), len(x)) of u values
        v : numpy
            ND array of shape (len(z), len(y), len(x)) of v values
        w : numpy
            ND array of shape (len(z), len(y), len(x)) of w values
        """

        x_coord= np.arange(self.bbox_ffield[bbox_id]['x']['min'],
                            self.bbox_ffield[bbox_id]['x']['max'] +
                            self.bbox_ffield[bbox_id]['x']['res'],
                            self.bbox_ffield[bbox_id]['x']['res'])

        y_coord= np.arange(self.bbox_ffield[bbox_id]['y']['min'],
                            self.bbox_ffield[bbox_id]['y']['max'] +
                            self.bbox_ffield[bbox_id]['y']['res'],
                            self.bbox_ffield[bbox_id]['y']['res'])

        z_coord= np.arange(self.bbox_ffield[bbox_id]['z']['min'],
                            self.bbox_ffield[bbox_id]['z']['max'] +
                            self.bbox_ffield[bbox_id]['z']['res'],
                            self.bbox_ffield[bbox_id]['z']['res'])

        base_array = np.empty((len(z_coord), len(y_coord),len(x_coord)))

        u = self._pl_fill_in(base_array, u)
        v = self._pl_fill_in(base_array, v)
        w = self._pl_fill_in(base_array, w)

        self.ffield = xr.Dataset({'u': (['z', 'y', 'x'], u),
                                  'v': (['z', 'y', 'x'], v),
                                  'w': (['z', 'y', 'x'], w)},
                                 coords={
                                         'x': x_coord,
                                         'y': y_coord,
                                         'z': z_coord,
                                         'Easting' : (['x'], x_coord),
                                         'Northing' : (['y'], y_coord),
                                         'Height' : (['z'], z_coord)
                                         })
        # Adding metadata
        self.ffield = self._add_metadata(self.ffield, metadata,
                                         'Flow field dataset')
        self.ffield.attrs['generator'] = bbox_id


    @staticmethod
    def _pl_fill_in(empty_array, values):
        """
        Fills empty array with repeated values

        Parameters
        ----------
        empty_array : numpy
            Numpy array of shape ...
        values : numpy
            1D array of values to be filled in empty array

        Returns
        -------
        numpy
            Populated array with values
        """

        full_array = np.copy(empty_array)
        for i, value in enumerate(values):
            full_array[i, :, :] = value
        return full_array


    def _upd8_meas_cfg(self, lidar_id, scan_type, az, el, rng, no_los,
                      no_scans, scn_speed, sectrsz, scn_tm, rtn_tm, max_speed, 
                      max_acc,scans_per_10min = None):
        """
        Updates measurement config

        Parameters
        ----------
        lidar_id : str
            ID of lidar for which measurement config is updates
        scan_type : str
            Scan type
        az : numpy
            Array of all azimuth positions
        el : numpy
            Array of all elevation positions
        rng : numpy
            Array of all range values
        no_los : int
            Number of line of sight
        no_scans : int
            Number of scans
        scn_speed : float
            Averaged scan speed through all measurement points
        sectrsz : float
            Size of scanned area (sector size for PPI)
        scn_tm : float
            Total time for scanning through all measurement points
        rtn_tm : float
            Time to return from the end to the begining of the scan
        max_speed : float
            Maximum permitted angular speed, in deg/s
        max_acc : float
            Maximum permitted angular acceleration, in deg/s^2
        scans_per_10min : int, optional
            Number of scans per 10 minutes, by default None
        """

        self.meas_cfg[lidar_id]['config'].update({'scan_type':scan_type})
        self.meas_cfg[lidar_id]['config'].update({'max_scn_speed':max_speed})
        self.meas_cfg[lidar_id]['config'].update({'max_scn_acc':max_acc})
        self.meas_cfg[lidar_id]['config'].update({'scn_speed':scn_speed})
        self.meas_cfg[lidar_id]['config'].update({'no_los':no_los})
        self.meas_cfg[lidar_id]['config'].update({'scans_per_10min':scans_per_10min})
        self.meas_cfg[lidar_id]['config'].update({'no_scans':no_scans})
        self.meas_cfg[lidar_id]['config'].update({'sectrsz':sectrsz})
        self.meas_cfg[lidar_id]['config'].update({'scn_tm':scn_tm})
        self.meas_cfg[lidar_id]['config'].update({'rtn_tm':rtn_tm})
        self.meas_cfg[lidar_id]['config'].update({'az':az})
        self.meas_cfg[lidar_id]['config'].update({'el':el})
        self.meas_cfg[lidar_id]['config'].update({'rng':rng})

    def _cr8_probing_ds(self, lidar_id, az, el, rng, time):
        """
        Creates Mocalum probing xr.DataSet

        Parameters
        ----------
        lidar_id : str
            ID of lidar for which probing dataset is created
        az : numpy
            Array of azimuth positions
        el : numpy
            Array of elevation positions
        rng : numpy
            Array of range values
        time : numpy
            Array of time values
        """
        # generating empty uncertainty and xyz arrays
        unc  = np.full(az.shape, 0.0, dtype=float)
        xyz  = np.full(az.shape, np.nan, dtype=float)

        # pulling information from measurement config dictionary
        s_sz = (self.meas_cfg[lidar_id]['config']['sectrsz']
                if 'sectrsz' in self.meas_cfg[lidar_id]['config'] else None)
        n_scn = (self.meas_cfg[lidar_id]['config']['no_scans']
                if 'no_scans' in self.meas_cfg[lidar_id]['config'] else None)
        no_los = (self.meas_cfg[lidar_id]['config']['no_los']
                if 'no_los' in self.meas_cfg[lidar_id]['config'] else None)
        s_tm = (self.meas_cfg[lidar_id]['config']['scn_tm']
                if 'scn_tm' in self.meas_cfg[lidar_id]['config'] else None)
        r_tm = (self.meas_cfg[lidar_id]['config']['rtn_tm']
                if 'rtn_tm' in self.meas_cfg[lidar_id]['config'] else None)
        lidar_pos = (self.meas_cfg[lidar_id]['position']
                if 'position' in self.meas_cfg[lidar_id] else None)


        probing_ds = xr.Dataset({'az': (['time'], az),
                                   'el': (['time'], el),
                                   'rng': (['time'], rng),
                                   'x': (['time'], xyz),
                                   'y': (['time'], xyz),
                                   'z': (['time'], xyz),
                                   'unc_az': (['time'], unc),
                                   'unc_el': (['time'], unc),
                                   'unc_rng': (['time'], unc),
                                   'unc_est': (['time'], unc),
                                   'sectrsz':(s_sz),
                                   'no_scans':(n_scn),
                                   'no_los':(no_los),
                                   'scan_tm':(s_tm),
                                   'return_tm':(r_tm),
                                   'lidar_pos_x':(lidar_pos[0]),
                                   'lidar_pos_y':(lidar_pos[1]),
                                   'lidar_pos_z':(lidar_pos[2]),
                                   },coords={'time': time})

        # adding/updating metadata
        probing_ds = self._add_metadata(probing_ds, metadata,
                                     'Lidar atmosphere probing dataset')

        self.probing.update({lidar_id:probing_ds})


    def _add_unc(self, lidar_id, unc_term, samples):
        """
        Adds generated uncertainties to probing xr.DataSet

        Parameters
        ----------
        lidar_id : str
            Id of lidar from self.meas_cfg lidar dict
        unc_term : str
            ID of uncertainty term
        samples : numpy
            Numpy array containing generated values for given uncertainty term
        """
        self.tmp_unc = unc_term
        self.tmp_unc_val = samples
        self.probing[lidar_id][unc_term].values  = samples


    def _add_xyz(self, lidar_id, x, y, z):
        """
        Adds Cartesian coordinates to probing dataset

        Parameters
        ----------
        lidar_id : str
            Id of lidar for which probing dataset is being updated
        x : numpy
            Array of x values
        y : numpy
            Array of y values
        z : numpy
            Array of z values
        """
        self.probing[lidar_id].x.values = x.values
        self.probing[lidar_id].y.values = y.values
        self.probing[lidar_id].z.values = z.values

    def _cr8_los_ds(self, lidar_id, los):
        """
        Create mocalum los xarray dataset

        Parameters
        ----------
        lidar_id : str
            Id of lidar for which LOS dataset is being created
        los : numpy
            Array of los speed
        """
        # TODO: detect what type of measurements it is (PPI, RHI, etc.)

        los = xr.Dataset({'vrad': (['time'], los),
                           'az': (['time'],  self.probing[lidar_id].az.values),
                           'el': (['time'],  self.probing[lidar_id].el.values),
                           'rng': (['time'], self.probing[lidar_id].rng.values),
                           'no_scans':(self.probing[lidar_id].no_scans.values),
                           'no_los':  (self.probing[lidar_id].no_los.values)},
                           coords={'time': self.probing[lidar_id].time.values})


        # adding/updating metadata
        los = self._add_metadata(los, metadata,'Radial wind speed dataset')
        self.los.update({lidar_id:los})

    def _cr8_sonic_ds(self, points_pos, time, u, v, w, ws, wdir):
        """
        Create mocalum virtual sonic anemometer xarray dataset

        Parameters
        ----------
        points_pos : numpy
            Measurement point position as (n,3) shaped numpy array
        time : numpy
            Numpy array of time instances at which sonic is 'measuring'
        u : numpy
            Array of reconstructed u values
        v : numpy
            Array of reconstructed v values
        ws : numpy
            Array of reconstructed wind speed values
        wdir : numpy
            Array of reconstructed wind direction values
        w : numpy, optional
            Array of reconstructed vertical wind speed, by default None
        """
        shape = points_pos.shape

        self.sonic_wind = xr.Dataset({'ws': (['time', 'point'], ws),
                                'wdir':(['time','point'], wdir),
                                'u': (['time', 'point'], u),
                                'v': ([ 'time', 'point'], v),
                                'w': (['time', 'point'], w)
                                },coords={'time': time,
                                          'point' : np.arange(1,len(points_pos)+1, 1),
                                          'x':(['point'], points_pos[:,0]),
                                          'y':(['point'], points_pos[:,1]),
                                          'z':(['point'], points_pos[:,2])})

        # adding/updating metadata
        self.sonic_wind = self._add_metadata(self.sonic_wind, metadata,
                                             'Virtual sonics')


    def _cr8_rc_wind_ds(self, scan_type, u, v, ws, wdir, w = None, availability = None):
        """
        Create mocalum reconstructed wind xarray dataset

        Parameters
        ----------
        scan_type : str
            Indicates scan type used to produce background LOS measurements
        u : numpy
            Array of reconstructed u values
        v : numpy
            Array of reconstructed v values
        ws : numpy
            Array of reconstructed wind speed values
        wdir : numpy
            Array of reconstructed wind direction values
        w : numpy, optional
            Array of reconstructed vertical wind speed, by default None
        """
        shape = ws.shape
        if type(availability) == type(None):
            availability = np.ones_like(ws)

        if type(w) != type(None):
            self.rc_wind = xr.Dataset({'ws': (['scan', 'point'], ws),
                                    'wdir':(['scan', 'point'], wdir),
                                    'u': (['scan', 'point'], u),
                                    'v': (['scan', 'point'], v),
                                    'w': (['scan', 'point'], w),
                                    'availability': (['scan', 'point'], availability)
                                    },coords={'scan': np.arange(1,shape[0]+1, 1),
                                              'point' : np.arange(1,shape[1]+1, 1)})
        else:
            self.rc_wind = xr.Dataset({'ws': (['scan','point'], ws),
                                    'wdir':(['scan', 'point'], wdir),
                                    'u': (['scan', 'point'], u),
                                    'v': (['scan', 'point'], v),
                                    'availability': (['scan', 'point'], availability)
                                    },coords={'scan': np.arange(1,shape[0]+1, 1),
                                              'point' : np.arange(1,shape[1]+1, 1)})


        # adding/updating metadata
        self.rc_wind = self._add_metadata(self.rc_wind, metadata,
                                      'Reconstructed wind')
        self.rc_wind.attrs['scan_type'] = scan_type


    def _get_ffield_coords(self, id):
        """
        Gets coordinates of flow field points

        Parameters
        ----------
        id : str
            BBOX cfg id

        Returns
        -------
        list
            list of numpy arrays for x, y, z, and time coordinates
        """
        bbox_cfg=self.bbox_ffield[id]

        x_coords = np.arange(bbox_cfg['x']['min'] -   bbox_cfg['x']['res'],
                             bbox_cfg['x']['max'] + 2*bbox_cfg['x']['res'],
                             bbox_cfg['x']['res'])

        y_coords = np.arange(bbox_cfg['y']['min'] -   bbox_cfg['y']['res'],
                             bbox_cfg['y']['max'] + 2*bbox_cfg['y']['res'],
                             bbox_cfg['y']['res'])

        z_coords = np.arange(bbox_cfg['z']['min'] -   bbox_cfg['z']['res'],
                             bbox_cfg['z']['max'] + 2*bbox_cfg['z']['res'],
                             bbox_cfg['z']['res'])

        t_coords = np.arange(bbox_cfg['t']['min'],
                             bbox_cfg['t']['max'] + bbox_cfg['t']['res'],
                             bbox_cfg['t']['res'])

        return x_coords, y_coords, z_coords, t_coords


    @staticmethod
    def _add_metadata(ds, metadata, ds_title=''):
        """
        Adds metadata to xr.DataSet

        Parameters
        ----------
        ds : xr.DataSet
            Mocalum xarray DataSet
        metadata : module
            Python module containing dictionaries of metadata
        ds_title : str, optional
            Title of DataSet, by default ''

        Returns
        -------
        xr.DataSet
            Mocalum xarray DataSet enriched with metadata
        """
        for var in ds.data_vars.keys():
            if var in metadata.VARS:
                ds[var].attrs = metadata.VARS[var]
        for coord in ds.coords.keys():
            if coord in metadata.DIMS:
                ds[coord].attrs = metadata.DIMS[coord]
        ds.attrs['title'] = ds_title
        return ds


data = Data()