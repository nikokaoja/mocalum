"""This module contains data class which stores data created in
the interaction of users with mocalum.
"""
import time
from . import metadata
import numpy as np
from numpy.linalg import inv as inv
import xarray as xr
from tqdm import tqdm


class Data:
    def __init__(self):
        self.temp_u = None
        self.temp_v = None
        self.probing = None
        self.los = None
        self.ffield = None
        self._ffield = None
        self.rc_wind = None
        self.fmodel_cfg = {}
        self.meas_cfg = {}
        self.ffield_bbox_cfg = {}
        self.unc_cfg = {'azimuth':{'mu':0, 'std':0.1},
                        'elevation':{'mu':0, 'std':0.1},
                        'range':{'mu':0, 'std':10},
                        'estimation':{'mu':0, 'std':0.1},
                        'corr_coef':0}

    def _cr8_fmodel_cfg(self, cfg):
        self.fmodel_cfg = cfg


    def _cr8_bbox_dict(self, CRS,
                       x_coord, y_coord, z_coord,t_coord,
                       x_offset, y_offset, z_offset,t_offset,
                       x_res, y_res, z_res, t_res):

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

        self.ffield_bbox_cfg = bbox_cfg



    def _cr8_3d_tfield_ds(self, turb_df):

        _ , y, z, t = self._get_ffield_coords()

        turb_np = turb_df.to_numpy().transpose().ravel()
        turb_np = turb_np.reshape(int(len(turb_np)/len(t)), len(t))

        self.temp_u = turb_np[0::3]
        self.temp_v = turb_np[1::3]



        # -1 to aligned properly axis
        R_tb = -self.ffield_bbox_cfg['CRS']['rot_matrix']

        # rotate u and v component to be Eastward and Northward wind
        # according to the met conventions
        uv = np.array([turb_np[0::3],turb_np[1::3]]).transpose()
        tmp_shape = uv.shape
        uv = uv.reshape(tmp_shape[0]*tmp_shape[1],2).dot(inv(R_tb)).reshape(tmp_shape)
        uv = uv.transpose()

        u = uv[0].reshape(len(y), len(z) ,len(t)).transpose(1,0,2)
        v = uv[1].reshape(len(y), len(z) ,len(t)).transpose(1,0,2)
        w = turb_np[2::3].reshape(len(y), len(z) ,len(t)).transpose(1,0,2)

        # before rotation
        # u = turb_np[0::3].reshape(len(y), len(z) ,len(t)).transpose(1,0,2)
        # v = turb_np[1::3].reshape(len(y), len(z) ,len(t)).transpose(1,0,2)
        # w = turb_np[2::3].reshape(len(y), len(z) ,len(t)).transpose(1,0,2)


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

    def _cr8_4d_tfield_ds(self, u, v, w, x, t):

        self._ffield = self.ffield
        R_tb = self.ffield_bbox_cfg['CRS']['rot_matrix']
        y = self.ffield.y.values
        z = self.ffield.z.values

        # # make Easting, Northing coordinates:
        # east_north = np.array([x,y]).transpose().dot(inv(R_tb))
        # east = east_north[:,0]
        # north = east_north[:,1]

        ew = np.empty((len(x),len(y),2))


        for i in range(0,len(x)):
            for j in range(0,len(y)):
                ew[i,j] = np.array([x[i],y[j]]).dot(inv(R_tb))


        self.ffield = xr.Dataset({'u': (['time', 'z', 'y', 'x'], u),
                                     'v': (['time', 'z', 'y', 'x'], v),
                                     'w': (['time', 'z', 'y', 'x'], w)},
                                coords={'time': t,
                                        'y': y,
                                        'z': z,
                                        'x': x,
                                         'Easting' : (['x','y'], ew[:,:,0]),
                                         'Northing' : (['x','y'], ew[:,:,1]),
                                         'Height' : (['z'], z)
                                        })

        self.ffield.attrs['generator'] = 'PyConTurb'
        self.ffield = self._add_metadata(self.ffield, metadata,
                                         'Turbulent flow field dataset')




    def _upd8_tfield_ds(self, turb_df):

        _ , y, z, t = self._get_ffield_coords()

        turb_np = turb_df.to_numpy().transpose().ravel()
        turb_np = turb_np.reshape(int(len(turb_np)/len(t)), len(t))

        u = turb_np[0::3].reshape(len(y), len(z) ,len(t)).transpose(1,0,2)
        v = turb_np[1::3].reshape(len(y), len(z) ,len(t)).transpose(1,0,2)
        w = turb_np[2::3].reshape(len(y), len(z) ,len(t)).transpose(1,0,2)


        base_array = np.empty((len(self.tfield.z),
                                len(self.tfield.y),
                                len(self.tfield.t)))

        self.tfield.u.values = self._pl_fill_in(base_array, u)
        self.tfield.v.values = self._pl_fill_in(base_array, v)
        self.tfield.w.values = self._pl_fill_in(base_array, w)
        self.tfield.attrs['generator'] = 'PyConTurb'


    def _cr8_plfield_ds(self, u, v, w):
        x_coord= np.arange(self.ffield_bbox_cfg['x']['min'],
                           self.ffield_bbox_cfg['x']['max'],
                           self.ffield_bbox_cfg['x']['res'])

        y_coord= np.arange(self.ffield_bbox_cfg['y']['min'],
                           self.ffield_bbox_cfg['y']['max'],
                           self.ffield_bbox_cfg['y']['res'])

        z_coord= np.arange(self.ffield_bbox_cfg['z']['min'],
                           self.ffield_bbox_cfg['z']['max'],
                           self.ffield_bbox_cfg['z']['res'])

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
                                         'Northing' : (['y'], x_coord),
                                         'Height' : (['z'], z_coord)
                                         })
        # Adding metadata
        self.ffield = self._add_metadata(self.ffield, metadata,
                                         'Flow field dataset')
        self.ffield.attrs['generator'] = 'power_law_model'

    def _upd8_ffield_ds(self, u, v, w, no_dim = 3):

        if no_dim == 3:
            # create base array
            base_array = np.empty((len(self.ffield.z),
                                   len(self.ffield.y),
                                   len(self.ffield.x)))


            self.ffield.u.values = self._pl_fill_in(base_array, u)
            self.ffield.v.values = self._pl_fill_in(base_array, v)
            self.ffield.w.values = self._pl_fill_in(base_array, w)
            self.ffield.attrs['generator'] = 'power_law_model'

    @staticmethod
    def _pl_fill_in(empty_array, values):

        full_array = np.copy(empty_array)
        for i, value in enumerate(values):
            full_array[i, :, :] = value
        return full_array


    def _cr8_meas_cfg(self, lidar_pos, scan_type, az, el, rng, no_los,
                      no_scans, scn_speed, sectrsz, rtn_tm, max_speed, max_acc):

        self.meas_cfg.update({'lidar_pos':lidar_pos})
        self.meas_cfg.update({'scan_type':scan_type})
        self.meas_cfg.update({'max_scn_speed':max_speed})
        self.meas_cfg.update({'max_scn_acc':max_acc})
        self.meas_cfg.update({'scn_speed':scn_speed})
        self.meas_cfg.update({'no_los':no_los})
        self.meas_cfg.update({'no_scans':no_scans})
        self.meas_cfg.update({'sectrsz':sectrsz})
        self.meas_cfg.update({'scn_tm':sectrsz*scn_speed})
        self.meas_cfg.update({'rtn_tm':rtn_tm})
        self.meas_cfg.update({'az':az})
        self.meas_cfg.update({'el':el})
        self.meas_cfg.update({'rng':rng})

    def _cr8_probing_ds(self, az, el, rng, time):
        # generating empty uncertainty and xyz arrays
        unc  = np.full(az.shape, 0.0, dtype=float)
        xyz  = np.full(az.shape, np.nan, dtype=float)

        # pulling information from measurement config dictionary
        s_sz = self.meas_cfg['sectrsz'] if 'sectrsz' in self.meas_cfg else None
        n_scn = self.meas_cfg['no_scans'] if 'no_scans' in self.meas_cfg else None
        no_los = self.meas_cfg['no_los'] if 'no_los' in self.meas_cfg else None
        s_tm = self.meas_cfg['scan_tm'] if 'scan_tm' in self.meas_cfg else None
        r_tm = self.meas_cfg['return_tm'] if 'return_tm' in self.meas_cfg else None
        lidar_pos = self.meas_cfg['lidar_pos'] if 'lidar_pos' in self.meas_cfg else None

        self.probing = xr.Dataset({'az': (['time'], az),
                                   'el': (['time'], el),
                                   'rng': (['time'], rng),
                                   'x': (['time'], xyz),
                                   'y': (['time'], xyz),
                                   'z': (['time'], xyz),
                                   'unc_az': (['time'],  unc),
                                   'unc_el': (['time'],  unc),
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
        self.probing = self._add_metadata(self.probing, metadata,
                                     'Lidar atmosphere probing dataset')

    def _add_unc(self, unc_term, samples):
        self.probing[unc_term].values  = samples

    def _add_xyz(self, x, y, z):
        self.probing.x.values = x.values
        self.probing.y.values = y.values
        self.probing.z.values = z.values

    def _cr8_los_ds(self, los):
        # TODO: detect what type of measurements it is (PPI, RHI, etc.)
        # TODO: we need somehow to
        self.los = xr.Dataset({'vrad': (['time'], los),
                              'az': (['time'], self.probing.az.values),
                              'el': (['time'], self.probing.el.values),
                              'rng': (['time'], self.probing.rng.values),
                              'no_scans':(self.probing.no_scans.values),
                              'no_los':  (self.probing.no_los.values)
                              },coords={'time': self.probing.time.values})


        # adding/updating metadata
        self.los = self._add_metadata(self.los, metadata,
                                      'Radial wind speed dataset')

    def _cr8_rc_wind_ds(self, u, v, ws):
        self.rc_wind = xr.Dataset({'ws': (['scan'], ws),
                                   'u': (['scan'], u),
                                   'v': (['scan'], v)
                                   },coords={'scan': np.arange(1,len(u)+1, 1)})


        # adding/updating metadata
        self.rc_wind = self._add_metadata(self.rc_wind, metadata,
                                      'Reconstructed wind')
        self.rc_wind.attrs['scan_type'] = self.meas_cfg['scan_type']

    def _get_ffield_coords(self):
        bbox_cfg=self.ffield_bbox_cfg

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
        for var in ds.data_vars.keys():
            if var in metadata.VARS:
                ds[var].attrs = metadata.VARS[var]
        for coord in ds.coords.keys():
            if coord in metadata.DIMS:
                ds[coord].attrs = metadata.DIMS[coord]
        ds.attrs['title'] = ds_title
        return ds

data = Data()