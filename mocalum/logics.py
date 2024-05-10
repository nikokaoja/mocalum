"""This module contains a class which encapsulates methods that create or
operate on data which are stored in persistance module.
"""

import numpy as np
from numpy.linalg import inv as inv
from scipy.stats import circmean, circstd
import xarray as xr

# import data instance
from .persistance import data

# import utils functions
from .utils import move2time, spher2cart, get_plaw_uvw, project2los, _rot_matrix
from .utils import sliding_window_slicing, bbox_pts_from_array, bbox_pts_from_cfg
from .utils import calc_mean_step, safe_execute, generate_beam_coords
from .utils import trajectory2displacement, displacement2time
from .utils import ivap_rc, DBS_rc, dd_rc_array, td_rc_array, gen_unc

# turbulence box generation tools
from pyconturb import gen_turb, gen_spat_grid
from pyconturb.wind_profiles import power_profile
from pyconturb.sig_models import iec_sig, constant_sig

class Mocalum:

    def __init__(self):
        self.data = data # this is an instance of Data() from persistance.py
        self.x_res = 25
        self.y_res = 25
        self.z_res = 5
        self.t_res = None
        self.turbbox_time = 600 # seconds = 10 min

    # Methods related to adding lidars to Mocalum object
    #
    @staticmethod
    def _is_lidar_pos(lidar_pos):
        """
        Validates if the lidar position is according to definition

        Parameters
        ----------
        lidar_pos : ndarray
            nD array containing data with `float` or `int` type
            corresponding to x, y and z coordinates of a lidar.
            nD array data are expressed in meters.

        Returns
        -------
            True / False
        """

        rules = [True, True, True]

        exec("try: rules[0] = type(lidar_pos).__module__ == np.__name__\nexcept: rules[0]=False")
        exec("try: rules[1] = len(lidar_pos.shape) == 1\nexcept: rules[1]=False")
        exec("try: rules[1] = lidar_pos.shape[0] == 3\nexcept: rules[2]=False")


        messages = ["Not numpy array",
                    "Lidar position must be one dimensional numpy array",
                    "Lidar position array must contain three elements"]
        if all(rules):
            return True
        else:
            for i, rule in enumerate(rules):
                if rule == False:
                    print(messages[i])
            return False

    def add_lidar(self, id, lidar_pos, unc_dict = None):
        """Adds a lidar instance to the measurement config

        Lidars can be add one at time. Currently only the instrument position
        in UTM coordinate system is supported.

        Parameters
        ----------
        instrument_id : str, required
            String which identifies instrument in the instrument dictionary.
        lidar_pos : numpy, required
            1D array containing data with `float` or `int` type corresponding
            to Northing, Easting and Height coordinates of the instrument.
            1D array data are expressed in meters.
        unc_dict : dict
            Dictionary containing values of uncertainty contributors,
            by default it is set to None,

        Note
        ----
        unc_dict can have following keys:
            - unc_est : float
                Uncertainty in estimating radial velocity from Doppler spectra.
            - unc_rng : float
                Uncertainty in detecting range at which atmosphere is probed.
            - unc_az : float
                Uncertainty in the beam steering for the azimuth angle.
            - unc_el : float
                Uncertainty in the beam steering for the elevation angle.
            - corr_coef : float
                Correlation coffecient of inputs

        In case when unc_dict is None default values will be set:
            unc_est =  0.1  (in m/s)
            unc_rng = 1 (in m)
            unc_az = 0.1 (deg)
            unc_el = 0.1 (deg)
            corr_coef = 0 (uncorrelated inputs)
        """
        if not(self._is_lidar_pos(lidar_pos)):
            raise ValueError("Lidar position is not properly provided")


        unc_elements = ['unc_est', 'unc_az','unc_el',
                        'unc_rng', 'corr_coef']

        lidar_dict = {id:{'position': lidar_pos,
                          'uncertainty':{
                              'unc_az':{'mu':0,'std':0.1,'units':'deg'},
                              'unc_el':{'mu':0, 'std':0.1, 'units':'deg'},
                              'unc_rng':{'mu':0, 'std':10, 'units':'m'},
                              'unc_est':{'mu':0, 'std':0.1, 'units':'m.s^-1'},
                              'corr_coef':0},
                          'config': {},
                                        }
                        }

        if unc_dict != None:
            for element in unc_elements:
                if element in unc_dict:
                    if element != 'corr_coef':
                        lidar_dict[id]['uncertainty'][element]['std'] = unc_dict[element]
                    else:
                        lidar_dict[id]['uncertainty'][element] = unc_dict[element]

        self.data.meas_cfg.update(lidar_dict)


    # Methods related to adding measurement points to Mocalum object
    #
    def _calc_xyz(self, lidar_id):
        """
        Calculates x,y and z coordinates of measurements points and updates
        probing xarray DataSet with the coordinates

        Parameters
        ----------
        lidar_id : str
            Id of lidar from self.data.meas_cfg lidar dict
        """
        probing_ds = self.data.probing[lidar_id]
        x,y,z = spher2cart(probing_ds.az  + probing_ds.unc_az,
                           probing_ds.el  + probing_ds.unc_el,
                           probing_ds.rng  + probing_ds.unc_rng)

        x.values+=self.data.meas_cfg[lidar_id]['position'][0]
        y.values+=self.data.meas_cfg[lidar_id]['position'][1]
        z.values+=self.data.meas_cfg[lidar_id]['position'][2]
        self.x = x
        self.data._add_xyz(lidar_id,x,y,z)

    def _cr8_bbox_meas_pts(self, lidar_id):
        """
        Creates 4D bounding box around measurement points of lidar

        Parameters
        ----------
        lidar_id : str
            Id of lidar from self.data.meas_cfg lidar dict
        """

        CRS = {'x':'Absolute coordinate, coresponds to Easting in m',
               'y':'Absolute coordinate, coresponds to Northing in m',
               'z':'Absolute coordinate, coresponds to height above sea level in m',
               'rot_matrix':_rot_matrix(90)}

        x_coord= np.array([self.data.probing[lidar_id].x.min() - self.x_res,
                           self.data.probing[lidar_id].x.max() + self.x_res])
        y_coord= np.array([self.data.probing[lidar_id].y.min() - self.y_res,
                           self.data.probing[lidar_id].y.max() + self.y_res])
        z_coord= np.array([self.data.probing[lidar_id].z.min() - self.z_res,
                           self.data.probing[lidar_id].z.max() + self.z_res])
        t_coord = self.data.probing[lidar_id].time.values
        if len(t_coord) > 1:
            t_res = calc_mean_step(t_coord)
        else:
            t_res = 1
        self.t_res = t_res

        # create/updated flow field bounding box config dict
        self.data._cr8_bbox_dict('lidar', lidar_id, CRS,
                                 x_coord, y_coord, z_coord, t_coord,
                                 0,0,0,0,
                                 self.x_res, self.y_res, self.z_res, t_res)


    def generate_PPI_scan(self, lidar_id, PPI_cfg):
        """
        Generate measurement points for PPI scan and create probing DataSet

        Parameters
        ----------
        lidar_id : str
            Id of lidar to be consider for genration of PPI measurement points
        PPI_cfg : dict
            Dictionary holding configuration of PPI scan

        Notes
        -----
        PPI_cfg must contain following keys having values in specific type:
            - sector_size : int
                 Size of the scanned sector in degrees
            - azimuth_mid : float, int
                 Central azimuth angle of the PPI scanned arc
            - angular_step : float, int
                 Incremental step performed to complete a PPI scan
            - acq_time : float
                 Acquisition time per angular step
            - elevation : float, int
                 Elevation angle of PPI scan
            - range : float, int
                 Range at which measurements should take place
            - no_scans : int, optional
                 Number of PPI scans must be equal or bigger than 1
            - max_speed : int
                 Max permitted angular speed
            - max_acc : int
                 Max permitted angular acceleration

        """

        if 'max_acc' not in PPI_cfg or 'max_speed' not in PPI_cfg:
            no_kinematics = True
        else:
            no_kinematics = False

        sector_size = PPI_cfg['sector_size']
        azimuth_mid = PPI_cfg['azimuth_mid']
        angular_step = PPI_cfg['angular_step']
        elevation = PPI_cfg['elevation']
        rng = PPI_cfg['range']
        no_scans  = PPI_cfg['no_scans']
        acq_time =  PPI_cfg['acq_time']
        scan_speed = angular_step / acq_time

        max_speed = PPI_cfg['max_speed']  if 'max_speed' in PPI_cfg else 50
        max_acc = PPI_cfg['max_acc']  if 'max_acc' in PPI_cfg else 100


        # prevent users stupidity raise errors!
        if angular_step <= 0:
            raise ValueError('Angular step must be > 0!')
        if no_scans <= 0:
            raise ValueError('Number of scans must be > 0!')
        if type(no_scans) is not int:
            raise ValueError('Number of scans must be int!')

        az = np.arange(azimuth_mid-sector_size/2,
                       azimuth_mid+sector_size/2,
                       angular_step, dtype=float)
        rng = np.full(len(az), rng, dtype=float)
        el = np.full(len(az), elevation, dtype=float)
        no_los = len(az)
        if no_kinematics:
            sweep_time = 0
        else:
            sweep_time = move2time(sector_size, max_acc, max_speed)
        scan_time = sector_size*scan_speed

        # lidar_pos = self.data.meas_cfg[lidar_id]['position']

        # create measurement configuration dictionary
        self.data._upd8_meas_cfg(lidar_id, 'PPI', az, el, rng, no_los, no_scans,
                                scan_speed, sector_size, no_los*scan_speed,sweep_time,
                                max_speed, max_acc)

        # tile probing coordinates to match no of scans
        az = np.tile(az, no_scans)
        rng = np.tile(rng, no_scans)
        el = np.tile(el, no_scans)

        # creating time dim
        time = np.arange(acq_time, (1 + sector_size/angular_step)*acq_time,
                         acq_time)

        # time = np.arange(0, sector_size*scan_speed + angular_step*scan_speed,
        #                 angular_step*scan_speed)
        time = np.tile(time, no_scans)
        to_add = np.repeat(scan_time + sweep_time, no_scans*no_los)
        multip = np.repeat(np.arange(0,no_scans), no_los)
        time = time + multip*to_add

        # create probing dataset later to be populated with probing unc
        self.data._cr8_probing_ds(lidar_id, az, el, rng, time)

        # adding xyz (these will not have uncertainties)
        self._calc_xyz(lidar_id)

        # # create flow field bounding box dict
        self._cr8_bbox_meas_pts(lidar_id)


    def generate_VAD_scan(self, lidar_id, VAD_cfg):
        """
        Generate measurement points for VAD scan (ground based lidar) and create probing DataSet

        Parameters
        ----------
        lidar_id : str
            Id of lidar to be consider for genration of PPI measurement points
        VAD_cfg : dict
            Dictionary holding configuration of VAD scan

        Notes
        -----
        VAD_cfg must contain following keys having values in specific type:
              lidar_model : str
                 'windcube_v1', 'windcube_v2', or 'ZX'. The lidar model determines 
                 the kind of wind field reconstruction, the number of beams, scan times etc.
              range : float, int
                 Range at which measurements should take place (distance above lidar, not LOS range)
              half_angle : float, int
                Angle between the beam (cone) and the vertical centre line 
              acq_time : float
                 Acquisition time per angular step
              no_10min_scans : int
                 Number of 10-min scan that should be simulated
                 
                 for ZX it must be specified how long a range is visited
        """

        no_10min_scans = VAD_cfg['no_10min_scans']
        half_angle = VAD_cfg['half_angle'] if 'half_angle' in VAD_cfg else 28
        rng = VAD_cfg['range']
        lidar_model = VAD_cfg['lidar_model']
        elevation = 90 - half_angle

        # prevent users stupidity and raise errors!
        if half_angle <= 0:
            raise ValueError('Opening angle between lines of sight must be > 0!')
        if no_10min_scans <= 0:
            raise ValueError('Number of scans must be > 0!')

        # windcube with 4 beams
        if lidar_model == 'windcube_v1':
            az = np.array([0, 90, 180, 270])
            ranges = np.full(len(az), rng/np.cos(half_angle/180*np.pi), dtype=float)
            el = np.full(len(az), elevation, dtype=float)
            acq_time = 1
            scan_time = 4 # no_LOS*scan_speed
            scans_per_10min = 150
            if not 'WFR_algorithm' in VAD_cfg:
                VAD_cfg.update({'WFR_algorithm':'DBS'})
            #        atmo_cfg.update({'flow_model':'PyConTurb'})
        # a windcube with 5 beams looks probably lke this:
        elif lidar_model == 'windcube_v2':
            az = np.array([0, 90, 180, 270, 0])
            ranges = np.full(len(az)-1, rng/np.cos(half_angle/180*np.pi), dtype=float)
            # translate the range into ranges along LOS;
            ranges = np.append(ranges, rng)
            # 5th beam measures straight up thus range above lidar is range along LOS
            el = np.full(len(az)-1, elevation, dtype=float)
            el = np.append(el, 90)
            acq_time = 1
            scan_time = 5 # no_LOS*scan_speed
            scans_per_10min = 120
            if not 'WFR_algorithm' in VAD_cfg:
                VAD_cfg.update({'WFR_algorithm':'DBS'})
        # a ZX probably looks like this:
        elif lidar_model == 'ZX':
            az = np.arange(0, 360, 7.2, dtype=float)
            # 50 scans correspond to 7.2 degree steps
            ranges = np.full(len(az), rng/np.cos(half_angle/180*np.pi), dtype=float)
            el = np.full(len(az), elevation, dtype=float)
            acq_time = 1/50
            scan_time = 1
            scans_per_10min = 600
            if not 'WFR_algorithm' in VAD_cfg:
                VAD_cfg.update({'WFR_algorithm':'VAD'})
        else:
            raise ValueError('lidar model unknown')

        no_los = len(az)
        no_scans = VAD_cfg['no_scans'] if 'no_scans' in VAD_cfg else scans_per_10min

        max_speed = 90 # degrees per second
        max_acc = 100
        sweep_time = 0  # no kinematics
        scan_speed = 1
        sector_size = 360

        # create measurement configuration dictionary
        self.data._upd8_meas_cfg(lidar_id, 'VAD', az, el, ranges, no_los, no_scans,
                                scan_speed, sector_size, scan_time, sweep_time,
                                max_speed, max_acc, scans_per_10min)

        # tile probing coordinates to match no of scans
        az = np.tile(az, no_scans)
        ranges = np.tile(ranges, no_scans)
        el = np.tile(el, no_scans)

        # creating time dim
        time = np.arange(acq_time, (1 + no_los)*acq_time,acq_time)

        time = np.tile(time, no_scans)
        to_add = np.repeat(scan_time + sweep_time, no_scans*no_los)
        multip = np.repeat(np.arange(0,no_scans), no_los)
        time = time + multip*to_add

        
        # create probing dataset later to be populated with probing unc
        self.data._cr8_probing_ds(lidar_id, az, el, ranges, time)

        # adding xyz (these will not have uncertainties)
        self._calc_xyz(lidar_id)

        # # create flow field bounding box dict
        self._cr8_bbox_meas_pts(lidar_id)
        

    def _single_lidar_ct_traj(self, lidar_id, CT_cfg):
        """
        Generate probing parameters for generate_complex_trajectory method

        Parameters
        ----------
        lidar_id : str
            Id of lidar to be consider for genration of PPI measurement points
        CT_cfg : dict
            Dictionary holding configuration of complex trajectory

        Notes
        -----
        CT_cfg must contain following keys having values in specific type:
            - points : numpy
                 ND array of x, y and z triplets of measurement points
            - no_scans : int, optional
                 Number of PPI scans must be equal or bigger than 1
            - max_speed : int, optional
                 Max permitted angular speed, by default 50 (deg/s)
            - max_acc : int, optional
                 Max permitted angular acceleration, by default 100 (deg/s^2)
            - acq_time : int, optional
                 Acquisition time to acquire LOS measurements, by default 1 (s)
            - sync : boolean, optional
                 Whether to synchronization or not trajectory among lidars,
                 by default True

        Returns
        -------
        dict
            Dictionary containing key-value pairs for azimuth, elevation,
            range, maximum angular displacement and trajectory timing
        """
        if 'max_acc' not in CT_cfg or 'max_speed' not in CT_cfg:
            no_kinematics = True
        else:
            no_kinematics = False

        meas_pts = CT_cfg['points']
        no_los = len(meas_pts)
        no_scans = CT_cfg['no_scans']  if 'no_scans' in CT_cfg else 1
        max_speed = CT_cfg['max_speed']  if 'max_speed' in CT_cfg else 50
        max_acc = CT_cfg['max_acc']  if 'max_acc' in CT_cfg else 100
        acq_time = CT_cfg['acq_time']  if 'acq_time' in CT_cfg else 1

        lidar_pos = self.data.meas_cfg[lidar_id]['position']
        beam_coord = generate_beam_coords(lidar_pos, meas_pts)

        az = beam_coord[:,0]
        el = beam_coord[:,1]
        rng = beam_coord[:,2]

        _ , _, displacement = trajectory2displacement(lidar_pos, meas_pts)
        max_displacement = np.max(displacement, axis = 1)
        if no_kinematics:
            traj_timing = np.full(no_los, 0)
        else:
            traj_timing = displacement2time(max_displacement,
                                            max_speed, max_acc)

        return {'az':az,
                'el':el,
                'rng':rng,
                'max_displacement':max_displacement,
                'traj_timing':traj_timing}

    def generate_complex_trajectory(self, lidar_id, CT_cfg):
        """
        Generate probing DataSet(s) for selected lidars and given measurement
        points


        Parameters
        ----------
        lidar_id : str, list or numpy
            Id(s) of lidar to be consider for generation of complex trajectory
        CT_cfg : dict
            Dictionary holding configuration of complex trajectory

        Notes
        -----
        CT_cfg must contain following keys having values in specific type:
            - points : numpy
                 ND array of x, y and z triplets of measurement points
            - no_scans : int, optional
                 Number of PPI scans must be equal or bigger than 1

        following keys are optional:
            - max_speed : int, optional
                 Max permitted angular speed, by default 50 (deg/s)
            - max_acc : int, optional
                 Max permitted angular acceleration, by default 100 (deg/s^2)
            - acq_time : int, optional
                 Acquisition time of LOS measurements, by default 1 (s)
            - sync : boolean, optional
                 Whether to synchronization or not trajectory among lidars,
                 by default True
        """
        #TODO: Implement sweeping trajectories in future

        meas_pts = CT_cfg['points']
        no_los = len(meas_pts)
        no_scans = CT_cfg['no_scans']  if 'no_scans' in CT_cfg else 1
        max_speed = CT_cfg['max_speed']  if 'max_speed' in CT_cfg else 50
        max_acc = CT_cfg['max_acc']  if 'max_acc' in CT_cfg else 100
        acq_time = CT_cfg['acq_time']  if 'acq_time' in CT_cfg else 1
        sync = CT_cfg['sync']  if 'sync' in CT_cfg else True
        sweep = CT_cfg['sweep']  if 'sweep' in CT_cfg else False

        # here we need loop on lidar_ids
        if type(lidar_id) == str:
            if lidar_id not in self.data.meas_cfg:
                raise ValueError('lidar id(s) does not exist !')
            else:
                lidar_id = [lidar_id]
        elif type(lidar_id) == list or type(lidar_id) == np.ndarray:
            for id in lidar_id:
                if id not in self.data.meas_cfg:
                    raise ValueError('lidar id(s) does not exist !')


        az = np.empty((len(lidar_id), len(meas_pts)))
        el = np.empty((len(lidar_id), len(meas_pts)))
        rng = np.empty((len(lidar_id), len(meas_pts)))
        max_displacement = np.empty((len(lidar_id), len(meas_pts)))
        traj_timing = np.empty((len(lidar_id), len(meas_pts)))


        for i, id in enumerate(lidar_id):

            info_dict = self._single_lidar_ct_traj(id, CT_cfg)
            az[i] = info_dict['az']
            el[i] = info_dict['el']
            rng[i] = info_dict['rng']
            max_displacement[i] = info_dict['max_displacement']
            traj_timing[i] = info_dict['traj_timing']

        max_time = traj_timing.max(axis = 0)

        for i in range(0,len(traj_timing)):
            traj_timing[i] = max_time if sync == True else traj_timing[i]

            if len(max_displacement[i]) == 1 or traj_timing[i][0]==0:
                avg_scan_speed = 0
            else:
                avg_scan_speed = np.mean(max_displacement[i]/traj_timing[i])
            sweep_time = traj_timing[i][0]
            traj_timing[i][0] = 0

            traj_timing[i] = np.cumsum(traj_timing[i] + acq_time)
            scan_time = traj_timing[i][-1]
            time = np.tile(traj_timing[i], no_scans)
            to_add = np.repeat(scan_time + sweep_time, no_scans*no_los)
            multip = np.repeat(np.arange(0,no_scans), no_los)
            time = time + multip*to_add

            # create measurement configuration dictionary
            self.data._upd8_meas_cfg(lidar_id[i], 'CT', az[i], el[i], rng[i],
                                     no_los, no_scans, avg_scan_speed,
                                     max_displacement[i].max(),
                                     traj_timing[i][-1], sweep_time, max_speed,
                                     max_acc)


            # tile probing coordinates to match no of scans
            az_tiled = np.tile(az[i], no_scans)
            rng_tiled = np.tile(rng[i], no_scans)
            el_tiled = np.tile(el[i], no_scans)

            # create probing dataset later to be populated with probing unc
            self.data._cr8_probing_ds(lidar_id[i], az_tiled,el_tiled,
                                      rng_tiled, time)

            # adding xyz (these will not have uncertainties)
            self._calc_xyz(lidar_id[i])

            # # create flow field bounding box dict
            self._cr8_bbox_meas_pts(lidar_id[i])


    # Methods related to generation of uncertainties
    #

    def generate_uncertainties(self, lidar_id, unc_cfg = None):
        """
        Generates uncertainty contributors to the radial wind speed uncertainty

        It is assumed that uncertainty contributions follow normal distribution.

        Parameters
        ----------
        lidar_id : str
            Id of lidar for which uncertainty contributors should be generated
        unc_cfg : dict, optional
            Config for method, by default None thus it takes existing config
            from self.data.meas_cfg
        """

        no_los = self.data.meas_cfg[lidar_id]['config']['no_los']
        no_scans = self.data.meas_cfg[lidar_id]['config']['no_scans']
        if unc_cfg == None:
            corr_coef = self.data.meas_cfg[lidar_id]['uncertainty']['corr_coef']
            unc_cfg = self.data.meas_cfg[lidar_id]['uncertainty'].copy()
            del unc_cfg['corr_coef']
        else:
            corr_coef = unc_cfg['corr_coef']
            unc_cfg = unc_cfg.copy()
            del unc_cfg['corr_coef']

        # samples uncertainty contributions considering normal distribution
        # and adds them to probing xr.DataSet
        for unc_term, cfg in unc_cfg.items():
            samples = gen_unc(np.full(no_los, cfg['mu']),
                              np.full(no_los, cfg['std']),
                              corr_coef, no_scans)
            self.data._add_unc(lidar_id, unc_term, samples.flatten())

        # update x,y,z coordinates of probing xr.DataSet considering
        # generated uncertainties
        self._calc_xyz(lidar_id)

        # update measurement points bounding box dict
        self._cr8_bbox_meas_pts(lidar_id)


    # Methods related to generation of flow field
    #
    def _create_absolute_bbox(self, lidar_id, bbox_id):
        """
        Creates wind field bounding box round measurement pts
        in absolute coordinate system

        Parameters
        ----------
        lidar_id : str, list, numpy
            ID of lidar(s) provided as str, list or np array of strings
        bbox_id : str
            ID of bounding box used to store it in self.data.bbox_ffield dict
        """
        linked_lidars = []
        try:
            if type(lidar_id) == str:
                linked_lidars += [lidar_id]
                bbox_meas_pts = self.data.bbox_meas_pts[lidar_id]

                z_min = bbox_meas_pts['z']['min']
                z_max = bbox_meas_pts['z']['max']
                z_res = bbox_meas_pts['z']['res']
                y_min = bbox_meas_pts['y']['min']
                y_max = bbox_meas_pts['y']['max']
                y_res = bbox_meas_pts['y']['res']
                x_min = bbox_meas_pts['x']['min']
                x_max = bbox_meas_pts['x']['max']
                x_res = bbox_meas_pts['x']['res']
                t_min = bbox_meas_pts['t']['min']
                t_max = bbox_meas_pts['t']['max']
                t_res = bbox_meas_pts['t']['res']

            elif type(lidar_id) == list or type(lidar_id) == np.ndarray:
                z_min = np.empty(len(lidar_id))
                z_max = np.empty(len(lidar_id))
                z_res = np.empty(len(lidar_id))
                y_min = np.empty(len(lidar_id))
                y_max = np.empty(len(lidar_id))
                y_res = np.empty(len(lidar_id))
                x_min = np.empty(len(lidar_id))
                x_max = np.empty(len(lidar_id))
                x_res = np.empty(len(lidar_id))
                t_min = np.empty(len(lidar_id))
                t_max = np.empty(len(lidar_id))
                t_res = np.empty(len(lidar_id))

                for i,id in enumerate(lidar_id):
                    linked_lidars += [id]
                    bbox_meas_pts = self.data.bbox_meas_pts[id]

                    z_min[i]= bbox_meas_pts['z']['min']
                    z_max[i]= bbox_meas_pts['z']['max']
                    z_res[i]= bbox_meas_pts['z']['res']
                    y_min[i]= bbox_meas_pts['y']['min']
                    y_max[i]= bbox_meas_pts['y']['max']
                    y_res[i]= bbox_meas_pts['y']['res']
                    x_min[i]= bbox_meas_pts['x']['min']
                    x_max[i]= bbox_meas_pts['x']['max']
                    x_res[i]= bbox_meas_pts['x']['res']
                    t_min[i]= bbox_meas_pts['t']['min']
                    t_max[i]= bbox_meas_pts['t']['max']
                    t_res[i]= bbox_meas_pts['t']['res']

                z_min = z_min.min()
                z_max = z_max.max()
                z_res = z_res.min()
                y_min = y_min.min()
                y_max = y_max.max()
                y_res = y_res.min()
                x_min = x_min.min()
                x_max = x_max.max()
                x_res = x_res.min()
                t_min = t_min.min()
                t_max = t_max.max()
                t_res = t_res.min()

            CRS = {'x':'Absolute coordinate, coresponds to Easting in m',
                   'y':'Absolute coordinate, coresponds to Northing in m',
                   'z':'Absolute coordinate, coresponds to height above sea level in m',
                   'rot_matrix':_rot_matrix(90)}

        # create/updated flow field bounding box config dict
            self.data._cr8_bbox_dict('ffield', bbox_id, CRS,
                                    np.array([x_min, x_max]),
                                    np.array([y_min, y_max]),
                                    np.array([z_min, z_max]),
                                    np.array([t_min, t_max]),
                                    0,0,0,0,
                                    x_res, y_res, z_res, t_res,
                                    linked_lidars=linked_lidars)

        except:
            raise ValueError('lidar id(s) does not exist !')

    def _get_prob_cords(self, lidar_id):
        """
        Gets probing coordinates from probing ds

        Parameters
        ----------
        lidar_id : str
            ID of lidar for which probing coordinates are extracted

        Returns
        -------
        numpy
            3D numpy array containing azimuth and elevation angles and range
        """

        x = self.data.probing[lidar_id].x.values
        y = self.data.probing[lidar_id].y.values
        z = self.data.probing[lidar_id].z.values

        return np.array([x,y,z]).transpose()

    def _create_relative_bbox(self, lidar_id, bbox_id,
                              wind_speed, wind_from_direction):
        """
        Creates wind field bounding box around measurement pts in a coordinate
        system aligned with wind direction

        Parameters
        ----------
        lidar_id : str, list, numpy
            ID of lidar(s) provided as str, list or np array of strings
        bbox_id : str
            ID of bounding box used to store it in self.data.bbox_ffield dict
        wind_speed : float
            Mean wind speed
        wind_from_direction : float
            Mean wind direction
        """
        # needs to find an average azimuth between n lidar
        linked_lidars = []
        try:
            if type(lidar_id) == str:
                avg_azimuth = self.data.meas_cfg[lidar_id]['config']['az'].mean()
                beam_xyz = self._get_prob_cords(lidar_id)
                linked_lidars += [lidar_id]

            elif type(lidar_id) == list or type(lidar_id) == np.ndarray:
                for i, id in enumerate(lidar_id):
                    linked_lidars += [id]
                    if i == 0:
                        beam_xyz =self._get_prob_cords(id)
                        aa = self.data.meas_cfg[id]['config']['az'].mean()
                    else:
                        beam_xyz = np.append(beam_xyz,
                                             self._get_prob_cords(id),
                                             axis = 0)
                        aa = np.append(aa, self.data.meas_cfg[id]['config']['az'].mean())
                avg_azimuth = aa.mean()
        except:
            raise ValueError('lidar id(s) does not exist !')

        # ws = kwargs['wind_speed']
        # wdir = kwargs['wind_from_direction']
        R_az = _rot_matrix(-avg_azimuth) #2D rot_matrix
        R_tb = _rot_matrix(wind_from_direction)
        CRS = {'x':'Relative coordinate, use inv(rot_matrix) to convert to abs',
               'y':'Relative coordinate, use inv(rot_matrix) to convert to abs',
               'z':'Absolute coordinate, coresponds to height above sea level',
               'rot_matrix':R_tb}


        # reproject probing coordinates to coordinate system where
        # y axis is aligned with the mean azimuth direction
        beam_xy= beam_xyz[:,(0,1)].dot(R_az)

        # create bounding box around the beam positions while rotating
        # the bounding box points back to absolute coordinate system
        bbox_pts = bbox_pts_from_array(beam_xy).dot(inv(R_az))

        # reproject those points now to turbulence box coordinate system
        # in which x axis is aligned with the mean wind direction:
        bbox_pts = bbox_pts.dot(R_tb)

        # Calculating time
        t_res = self.x_res / wind_speed
        self.t_res = t_res
        t_coord = np.arange(0, self.turbbox_time + t_res, t_res)

        self.data._cr8_bbox_dict('ffield', bbox_id, CRS,
                                 bbox_pts[:,0], bbox_pts[:,1], beam_xyz[:,2],
                                 t_coord,
                                 0, 0, 0, 0,
                                 self.x_res, self.y_res, self.z_res, t_res,
                                 linked_lidars = linked_lidars)


    def _power_law_ffield(self,lidar_id, atmo_cfg):
        """
        Generate uniform flow field which wind_speed varies only with height

        This method use power law expression for height dependent wind speed

        Parameters
        ----------
        lidar_id : str, list, numpy
            ID of lidar(s) provided as str, list or np array of strings
        atmo_cfg : dict
            Dictionary containing atmospheric parameters

        Notes
        -----
        atmo_cfg must contain the following keys:
         - "wind_speed"
         - "upward_velocity"
         - "wind_from_direction"
         - "reference_height"
         - "shear_exponent"

        Here upward_velocity correspond to vertical wind speed,
        wind_from_direction corresponds to wind direction, reference_height
        corresponds to the height at which wind_speed is provided, and
        shear_exponent indicates how wind_speed varies with height.
        """
        # updates atmo_cfg with type of flow_model used to generate u,v,w
        atmo_cfg.update({'flow_model':'power_law'})

        # updates/creates flow field config in self.data.fmodel_cfg
        self.data._cr8_fmodel_cfg(atmo_cfg)

        # updates/creates flow field bounding box in self.data.bbox_ffield
        self._create_absolute_bbox(lidar_id, 'power_law')

        # generate u,v, and w
        z_cfg = self.data.bbox_ffield['power_law']['z']
        u, v, w = get_plaw_uvw(np.arange(z_cfg['min'],
                                         z_cfg['max'] + z_cfg['res'],
                                         z_cfg['res']),
                               atmo_cfg['reference_height'],
                               atmo_cfg['wind_speed'],
                               atmo_cfg['upward_velocity'],
                               atmo_cfg['wind_from_direction'],
                               atmo_cfg['shear_exponent'])

        self.data._cr8_plfield_ds('power_law', u, v, w)

    def _turbulent_ffield(self,lidar_id, atmo_cfg, turb_parameter=None):
        """
        Generates turbulent flow field

        This method assumes power law expression for height dependent wind speed

        Parameters
        ----------
        lidar_id : str, list, numpy
            ID of lidar(s) provided as str, list or np array of strings
        atmo_cfg : dict
            Dictionary containing atmospheric parameters

        Notes
        -----
        atmo_cfg must contain the following keys:
         - "wind_speed"
         - "upward_velocity"
         - "wind_from_direction"
         - "reference_height"
         - "shear_exponent"

        Here upward_velocity correspond to vertical wind speed,
        wind_from_direction corresponds to wind direction, reference_height
        corresponds to the height at which wind_speed is provided, and
        shear_exponent indicates how wind_speed varies with height.
        """


        atmo_cfg.update({'flow_model':'PyConTurb'})
        self.data._cr8_fmodel_cfg(atmo_cfg)

        self._create_relative_bbox(lidar_id, 'turbulence_box',
                                   atmo_cfg['wind_speed'],
                                   atmo_cfg['wind_from_direction'])


        _, y, z, _ = self.data._get_ffield_coords('turbulence_box')
        spat_df = gen_spat_grid(y, z)
        T_tot = self.turbbox_time
        T_res = self.data.bbox_ffield['turbulence_box']['t']['res']

        if turb_parameter == None:
            sig_func = iec_sig
            kwargs = {'turb_class': 'C'}
        elif 'TI' in turb_parameter:
            sig_func = constant_sig
            TI = turb_parameter['TI']
            wind = atmo_cfg['wind_speed']
            kwargs = {'sig_vals': [TI*wind,TI*wind*0.8,TI*wind*0.5],
                      'comps': [0,1,2]}
        elif 'IEC_class' in turb_parameter:
            sig_func = iec_sig
            kwargs = {'turb_class': turb_parameter['IEC_class']}

        turb_df = gen_turb(spat_df, T=T_tot + T_res,
                           dt=T_res,
                           wsp_func = power_profile,
                           u_ref=atmo_cfg['wind_speed'],
                           z_ref=atmo_cfg['reference_height'],
                           alpha=atmo_cfg['shear_exponent'],
                           sig_func=sig_func,
                           **kwargs)

        # first create 3D ds
        self.data._cr8_3d_tfield_ds('turbulence_box', turb_df)

        # then from 3D ds make 4D
        self.data._cr8_4d_tfield_ds('turbulence_box')



    def generate_flow_field(self, lidar_id, atmo_cfg=None, flow_type="uniform", turb_parameter=None):
        """
        Generate flow field that entails measurement points of lidars

        Parameters
        ----------
        lidar_id : str, list, numpy
            ID of lidar(s) provided as str, list or np array of strings
        atmo_cfg : dict, optional
            dictionary containing atmospheric parameters, by default None
        flow_type : str, optional
            Flow field tyoe to be generated, it can be "uniform" or "turbulent",
            by default set to "uniform"

        Notes
        -----
        If atmo_cfg is provided it must have following keys:
         - "wind_speed"
         - "upward_velocity"
         - "wind_from_direction"
         - "reference_height"
         - "shear_exponent"

        Here upward_velocity correspond to vertical wind speed,
        wind_from_direction corresponds to wind direction, reference_height
        corresponds to the height at which wind_speed is provided, and
        shear_exponent indicates how wind_speed varies with height.
        """

        # make default atmo_cfg if atmo_cfg == None
        if atmo_cfg == None:
            atmo_cfg={'wind_speed':10,
            'upward_velocity':0,
            'wind_from_direction':180,
            'reference_height':100,
            'shear_exponent':0.2,
            }
        elif missing_keys := [key for key in ("wind_speed",
                                              "upward_velocity",
                                              "wind_from_direction",
                                              "reference_height",
                                              "shear_exponent",) if key not in atmo_cfg]:
                raise KeyError(f'Missing keys in atmo_cfg: {", ".join(missing_keys)}')

        if flow_type == "uniform":
            self._power_law_ffield(lidar_id, atmo_cfg)
        elif flow_type == "turbulent":
            self._turbulent_ffield(lidar_id, atmo_cfg, turb_parameter)

    # Methods related to projection of flow field to measurement points
    #
    def project_to_los(self, lidar_id):
        """Projects wind vector on laser beam line-of-sight

        Parameters
        ----------
        lidar_id : str
            ID of lidar on which LOS wind vector is being projected
        """
        ffield_id = self.data.ffield.generator

        if not(ffield_id in self.data.bbox_ffield):
            raise ValueError('Flow field does not exist')

        if type(lidar_id) == str:
            if not(lidar_id in self.data.bbox_ffield[ffield_id]['linked_lidars']):
                raise ValueError('lidar id not linked to flow field!')

        elif type(lidar_id) == list or type(lidar_id) == np.ndarray:
            for i,id in enumerate(lidar_id):
                if not(id in self.data.bbox_ffield[ffield_id]['linked_lidars']):
                    raise ValueError('lidar id not linked to flow field!')
        else:
            raise TypeError('Input type for lidar_id is not valid')


        if self.data.fmodel_cfg['flow_model'] == 'power_law':
            hmeas = self.data.probing[lidar_id].z.values
            u = self.data.ffield.u.isel(x=0,y=0).interp(z=hmeas)
            v = self.data.ffield.v.isel(x=0,y=0).interp(z=hmeas)
            w = self.data.ffield.w.isel(x=0,y=0).interp(z=hmeas)

        elif self.data.fmodel_cfg['flow_model'] == 'PyConTurb':

            # Normalize time to re-feed the turbulence
            time_prob = self.data.probing[lidar_id].time.values
            time_tbox = self.data.ffield.time.values
            time_norm = np.mod(time_prob, time_tbox.max())

            # Rotate relative coordinates to absolute coordinates
            R_tb = self.data.bbox_ffield[ffield_id]['CRS']['rot_matrix']
            x_r = self.data.probing[lidar_id].x.values
            y_r = self.data.probing[lidar_id].y.values
            z_coords = self.data.probing[lidar_id].z.values
            xy = np.array([x_r,y_r]).T.dot(R_tb)
            x_coords = xy[:,0]
            y_coords = xy[:,1]

            # Preparing input for xarray interpolation
            t = xr.DataArray(time_norm, dims='pt')
            x = xr.DataArray(x_coords, dims='pt')
            y = xr.DataArray(y_coords, dims='pt')
            z = xr.DataArray(z_coords, dims='pt')

            # Interpolation of 4D dataset to coordinates of interest
            ffield_pts = self.data.ffield.interp(time=t,
                                                 x=x,
                                                 y=y,
                                                 z=z, method='linear')
            # Extracts u, v and w from the derived xarray ds
            u = ffield_pts.u
            v = ffield_pts.v
            w = ffield_pts.w

        # Projects wind vector on LOS
        scan_type = self.data.meas_cfg[lidar_id]['config']['scan_type']
        ignore_elevation = True if scan_type=='PPI' else False
        los = project2los(u,v,w,
                        self.data.probing[lidar_id].az.values +
                        self.data.probing[lidar_id].unc_az.values,
                        self.data.probing[lidar_id].el.values +
                        self.data.probing[lidar_id].unc_el.values,
                        ignore_elevation)

        # Adds radial velocity estimation uncertainty
        los += self.data.probing[lidar_id].unc_est.values

        # Stores derived result as xr.DataSet
        self.data._cr8_los_ds(lidar_id, los)


    # Methods related to reconstruction of wind vector from los measurements
    @staticmethod
    def _scan_average(a, no_los, no_scans, no_avg):
        a_avg = np.nanmean(a.values.reshape(no_scans,no_los).reshape(int(no_scans/no_avg),
                                                          no_avg,
                                                          no_los),axis=1)
        return a_avg.flatten()


    def remove_samples(self, lidar_id, scan_config):
        if 'availability' in scan_config:
            availability = scan_config['availability']
        else:
            availability = 1
        if not hasattr(self.data,'los'):
            raise ValueError('The wind speed is not yet projected to the LOS.')
        if availability < 0 or availability > 1:
            raise ValueError('availability should be between zero and one (inclusive).')
        
        self.data.los[lidar_id].vrad[np.random.rand(
            *self.data.los[lidar_id].vrad.shape) >= availability] = np.nan


    def _IVAP_reconstruction(self, lidar_id, no_scans_avg):
        """
        IVAP wind reconstruction method

        Parameters
        ----------
        lidar_id : str
            Id of lidar for which LOS speed should be converted to wind vector

        no_scans_avg : int
            Number of scans to average prior reconstruction
        """

        # need first to average scans prior splitting for reconstruction

        no_los = self.data.meas_cfg[lidar_id]['config']['no_los']
        no_scans = self.data.meas_cfg[lidar_id]['config']['no_scans']

        if no_scans_avg != None and no_scans % no_scans_avg != 0:
            raise ValueError('Total number of scans must be divisible with number of scans to average')

        if type(no_scans_avg) == int and (no_scans % no_scans_avg == 0):
            no_scans_new = int(no_scans/no_scans_avg)
            az = self._scan_average(self.data.los[lidar_id].az, no_los,
                                   no_scans, no_scans_avg).reshape(no_scans_new, no_los)
            el = self._scan_average(self.data.los[lidar_id].el, no_los,
                                   no_scans, no_scans_avg).reshape(no_scans_new, no_los)
            los = self._scan_average(self.data.los[lidar_id].vrad, no_los,
                                   no_scans, no_scans_avg).reshape(no_scans_new, no_los)
            no_scans = int(no_scans/no_scans_avg)

        else:
            az = self.data.los[lidar_id].az.values.reshape(no_scans, no_los)
            el = self.data.los[lidar_id].el.values.reshape(no_scans, no_los)
            los = self.data.los[lidar_id].vrad.values.reshape(no_scans, no_los)


        u, v, ws, wdir = ivap_rc(los, az, 1)

        self.data._cr8_rc_wind_ds('single-Doppler IVAP',
                                  u.reshape(no_scans, 1),
                                  v.reshape(no_scans, 1),
                                  ws.reshape(no_scans, 1),
                                  wdir.reshape(no_scans, 1))


    def _DBS_reconstruction(self, lidar_id, no_scans_avg):
        """
        reconstruction of Windcube v1 and v2 goes here

        Parameters
        ----------
        lidar_id : TYPE
            DESCRIPTION.
        no_scans_avg : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        no_los = self.data.meas_cfg[lidar_id]['config']['no_los']
        no_scans = self.data.meas_cfg[lidar_id]['config']['no_scans']
        
        if type(no_scans_avg) == int and (no_scans % no_scans_avg != 0):
            raise ValueError('Total number of scans must be divisible with number of scans to average')

        # vector averaging approach
        if type(no_scans_avg) == str and no_scans_avg == 'vector':
            availability = (~np.isnan(self.data.los[lidar_id].vrad.values)).sum()/no_scans
            no_scans_avg = int(self.data.meas_cfg[lidar_id]['config']['scans_per_10min'])
            no_scans_new = int(no_scans/no_scans_avg)
            az = self._scan_average(self.data.los[lidar_id].az, no_los,
                                    no_scans, no_scans_avg).reshape(no_scans_new, no_los)
            el = self._scan_average(self.data.los[lidar_id].el, no_los,
                                    no_scans, no_scans_avg).reshape(no_scans_new, no_los)
            los = self._scan_average(self.data.los[lidar_id].vrad, no_los,
                                     no_scans, no_scans_avg).reshape(no_scans_new, no_los)
            #zenith = 90 - el
            u, v, ws, wdir = DBS_rc(los, az)
            no_scans = no_scans_new
        else:  # reconstruction every second
            az = self.data.los[lidar_id].az.values.reshape(no_scans, no_los)
            el = self.data.los[lidar_id].el.values.reshape(no_scans, no_los)
            los = self.data.los[lidar_id].vrad.values.reshape(no_scans, no_los)

            # in order to implement the continuous 'sliding window' output of the lidar 
            # copy the data shifted by one into the third diemnsion of the array 
            # and flatten in back to two dimensions:
            az_sliding = np.empty((no_los,no_scans,no_los))
            los_sliding = np.empty((no_los,no_scans,no_los))
            #el_sliding = np.empty((no_los,no_scans,no_los))
            for i in range(no_los):
                az_sliding[i,:,:] = np.roll(az, -i)
                los_sliding[i,:,:] = np.roll(los, -i)
                #el_sliding[i,:,:] = np.roll(el, -i)

            az_sliding = np.reshape(az_sliding, (no_scans*no_los, no_los), order='F')
            los_sliding = np.reshape(los_sliding, (no_scans*no_los, no_los), order='F')
            #el_sliding = np.reshape(el_sliding, (no_scans*no_los, no_los), order='F')
            #zenith = 90 - el_sliding
            u, v, ws, wdir = DBS_rc(los_sliding, az_sliding)
            availability = (~np.isnan(ws)).astype(int)
            no_scans = no_scans*no_los;

            if type(no_scans_avg) == str and no_scans_avg == 'scalar':
                no_avg = 600
                u = np.nanmean(u.reshape(int(no_scans/no_avg),no_avg),axis=1)
                v = np.nanmean(v.reshape(int(no_scans/no_avg),no_avg),axis=1)
                ws_tmp = ws.reshape(int(no_scans/no_avg),no_avg)
                availability = (~np.isnan(ws_tmp)).sum(axis=1)/no_avg
                ws = np.nanmean(ws_tmp,axis=1)
                wdir = circmean(wdir.reshape(int(no_scans/no_avg),no_avg),high=360,axis=1,nan_policy='omit')
                no_scans = int(no_scans/no_avg)


        self.data._cr8_rc_wind_ds('ground based lidar DBS',
                                  u.reshape(no_scans, 1),
                                  v.reshape(no_scans, 1),
                                  ws.reshape(no_scans, 1),
                                  wdir.reshape(no_scans, 1),
                                  None, availability.reshape(no_scans, 1))


    def _VAD_reconstruction(self, lidar_id, no_scans_avg):
        """
        Reconstruction of ZX goes here

        Parameters
        ----------
        lidar_id : TYPE
            DESCRIPTION.
        no_scans_avg : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        raise NotImplementedError('VAD reconstruction is not yet implemented')


    def _is_dual_Doppler(self, lidar_id):
        """
        Checks if provided lidar ids are valid prior reconstruction

        Parameters
        ----------
        lidar_id : list, numpy
            ID of two lidars provided as list or np array of strings
        """
        if not(type(lidar_id) == list or type(lidar_id) == np.ndarray):
            raise TypeError('lidar_id must be list or 1D numpy array!')
        elif not(len(lidar_id) == 2):
            raise ValueError('lidar_id must have two string elements')
        cfg = self.data.meas_cfg
        for id in lidar_id:
            if id not in cfg:
                raise ValueError('lidar id(s) does not exist!')

        total_los_1 = cfg[lidar_id[0]]['config']['no_los'] * cfg[lidar_id[0]]['config']['no_scans']
        total_los_2 = cfg[lidar_id[1]]['config']['no_los'] * cfg[lidar_id[1]]['config']['no_scans']

        if total_los_1!=total_los_2:
            raise ValueError('This is not dual-Doppler setup')


    def _is_triple_Doppler(self, lidar_id):
        """
        Checks if provided lidar ids are valid prior reconstruction

        Parameters
        ----------
        lidar_id : list, numpy
            ID of three lidars provided as list or np array of strings
        """
        if not(type(lidar_id) == list or type(lidar_id) == np.ndarray):
            raise TypeError('lidar_id must be list or 1D numpy array!')
        elif not(len(lidar_id) == 3):
            raise ValueError('lidar_id must have two string elements')
        cfg = self.data.meas_cfg
        for id in lidar_id:
            if id not in cfg:
                raise ValueError('lidar id(s) does not exist!')

        total_los_1 = cfg[lidar_id[0]]['config']['no_los'] * cfg[lidar_id[0]]['config']['no_scans']
        total_los_2 = cfg[lidar_id[1]]['config']['no_los'] * cfg[lidar_id[1]]['config']['no_scans']
        total_los_3 = cfg[lidar_id[2]]['config']['no_los'] * cfg[lidar_id[2]]['config']['no_scans']

        if total_los_1!=total_los_2:
            raise ValueError('This is not triple-Doppler setup')


    def _dual_Doppler_reconstruction(self, lidar_id, no_scans_avg):
        """
        Dual Doppler reconstruction of wind vector

        Parameters
        ----------
        lidar_id : list, numpy
            ID of two lidars provided as list or np array of strings
        no_scans_avg : int, optional
            Number of complete scans to average prior reconstruction,
            by default None reconstruction without averaging
        """

        self._is_dual_Doppler(lidar_id)
        no_los = self.data.meas_cfg[lidar_id[0]]['config']['no_los']
        no_scans = self.data.meas_cfg[lidar_id[0]]['config']['no_scans']

        if no_scans_avg != None and no_scans % no_scans_avg != 0:
            raise ValueError('Total number of scans must be divisible with number of scans to average')

        if type(no_scans_avg) == int and (no_scans % no_scans_avg == 0):
            az = np.empty((2, int(no_los*no_scans/no_scans_avg)))
            el = np.empty((2, int(no_los*no_scans/no_scans_avg)))
            los = np.empty((2, int(no_los*no_scans/no_scans_avg)))

            for i,id in enumerate(lidar_id):
                az[i] = self._scan_average(self.data.los[id].az, no_los,
                                           no_scans, no_scans_avg)
                el[i] = self._scan_average(self.data.los[id].el, no_los,
                                           no_scans, no_scans_avg)
                los[i] = self._scan_average(self.data.los[id].vrad, no_los,
                                           no_scans, no_scans_avg)

            no_scans = int(no_scans/no_scans_avg)

        else:
            az = np.empty((2, no_los*no_scans))
            el = np.empty((2, no_los*no_scans))
            los = np.empty((2, no_los*no_scans))

            for i,id in enumerate(lidar_id):
                self.az = self.data.los[id].az
                az[i] = self.data.los[id].az
                el[i] = self.data.los[id].el
                los[i] = self.data.los[id].vrad


        u, v, ws, wdir = dd_rc_array(los, az, el, 1)
        self.ws = ws.reshape(no_scans, no_los)

        self.data._cr8_rc_wind_ds('dual-Doppler CT',
                                  u.reshape(no_scans, no_los),
                                  v.reshape(no_scans, no_los),
                                  ws.reshape(no_scans, no_los),
                                  wdir.reshape(no_scans, no_los))




    def generate_virtual_sonic(self, meas_pts, time_steps):
        """
        Creates virtual sonic measurements at predefined locations and time steps

        Parameters
        ----------
        meas_pts : numpy
            Measurement point position as (n,3) shaped numpy array
        time_steps : numpy
            Numpy array of time instances at which sonic is 'measuring'
        """

        if self.data.fmodel_cfg['flow_model'] == 'power_law':
            hmeas = meas_pts[:,2]
            u = self.data.ffield.u.isel(x=0,y=0).interp(z=hmeas).values
            v = self.data.ffield.v.isel(x=0,y=0).interp(z=hmeas).values
            w = self.data.ffield.w.isel(x=0,y=0).interp(z=hmeas).values
            return print('No need for virtual sonic, this is uniform flow field!')

        elif self.data.fmodel_cfg['flow_model'] == 'PyConTurb':

            # Normalize time to re-feed the turbulence
            time_tbox = self.data.ffield.time.values
            time_norm = np.mod(time_steps, time_tbox.max())

            # Rotate absolute coordinates to relative coordinates
            R_tb = self.data.bbox_ffield['turbulence_box']['CRS']['rot_matrix']
            x_r = meas_pts[:,0]
            y_r = meas_pts[:,1]
            z_coords = meas_pts[:,2]
            xy = np.array([x_r,y_r]).T.dot(R_tb)
            x_coords = xy[:,0]
            y_coords = xy[:,1]

            # Preparing input for xarray interpolation
            t = xr.DataArray(np.tile(time_norm, len(x_coords)), dims='pt')
            x = xr.DataArray(np.repeat(x_coords, len(time_norm)), dims='pt')
            y = xr.DataArray(np.repeat(y_coords, len(time_norm)), dims='pt')
            z = xr.DataArray(np.repeat(z_coords, len(time_norm)), dims='pt')

            # Interpolation of 4D dataset to coordinates of interest
            ffield_pts = self.data.ffield.interp(time=t,
                                                 x=x,
                                                 y=y,
                                                 z=z, method='linear')
            # Extracts u, v and w from the derived xarray ds
            u = ffield_pts.u.values
            v = ffield_pts.v.values
            w = ffield_pts.w.values

        u = u.reshape(len(meas_pts), len(time_steps)).T
        v = v.reshape(len(meas_pts), len(time_steps)).T
        w = w.reshape(len(meas_pts), len(time_steps)).T


        ws = np.sqrt(u**2 + v**2)
        wdir = (90 - np.arctan2(-v,-u)* (180 / np.pi)) % 360

        self.data._cr8_sonic_ds(meas_pts, time_steps, u, v, w, ws, wdir)


    def _triple_Doppler_reconstruction(self, lidar_id, no_scans_avg):
        """
        Triple Doppler reconstruction of wind vector

        Parameters
        ----------
        lidar_id : list, numpy
            ID of three lidars provided as list or np array of strings
        no_scans_avg : int, optional
            Number of complete scans to average prior reconstruction,
            by default None reconstruction without averaging
        """

        self._is_triple_Doppler(lidar_id)
        no_los = self.data.meas_cfg[lidar_id[0]]['config']['no_los']
        no_scans = self.data.meas_cfg[lidar_id[0]]['config']['no_scans']

        if no_scans_avg != None and no_scans % no_scans_avg != 0:
            raise ValueError('Total number of scans must be divisible with number of scans to average')

        if type(no_scans_avg) == int and (no_scans % no_scans_avg == 0):
            az = np.empty((3, int(no_los*no_scans/no_scans_avg)))
            el = np.empty((3, int(no_los*no_scans/no_scans_avg)))
            los = np.empty((3, int(no_los*no_scans/no_scans_avg)))

            for i,id in enumerate(lidar_id):
                self.az = self.data.los[id].az
                az[i] = self._scan_average(self.data.los[id].az, no_los,
                                           no_scans, no_scans_avg)
                el[i] = self._scan_average(self.data.los[id].el, no_los,
                                           no_scans, no_scans_avg)
                los[i] = self._scan_average(self.data.los[id].vrad, no_los,
                                           no_scans, no_scans_avg)

            no_scans = int(no_scans/no_scans_avg)

        else:
            az = np.empty((3, no_los*no_scans))
            el = np.empty((3, no_los*no_scans))
            los = np.empty((3, no_los*no_scans))

            for i,id in enumerate(lidar_id):
                az[i] = self.data.los[id].az
                el[i] = self.data.los[id].el
                los[i] = self.data.los[id].vrad

        # here we should introduce averaging according to no of scans

        u, v, w, ws, wdir = td_rc_array(los, az, el)

        self.data._cr8_rc_wind_ds('triple-Doppler CT',
                                  u.reshape(no_scans, no_los),
                                  v.reshape(no_scans, no_los),
                                  ws.reshape(no_scans, no_los),
                                  wdir.reshape(no_scans, no_los),
                                  w.reshape(no_scans, no_los))


    def reconstruct_wind(self, lidar_id, rc_method = 'IVAP', no_scans_avg = None):
        """Reconstructs wind speed according to the selected retrieval method

        Parameters
        ----------
        lidar_id : str
            Id of lidar for which LOS speed should be converted to wind vector
        rc_method : str, optional
            Retrieval method, by default 'IVAP'
        """

        if rc_method == 'IVAP':
            self._IVAP_reconstruction(lidar_id,no_scans_avg)
        elif rc_method == 'dual-Doppler':
            self._dual_Doppler_reconstruction(lidar_id,no_scans_avg)
        elif rc_method == 'triple-Doppler':
            self._triple_Doppler_reconstruction(lidar_id,no_scans_avg)
        elif rc_method ==  'DBS':
            self._DBS_reconstruction(lidar_id, no_scans_avg)
        elif rc_method == 'VAD':
            self._VAD_reconstruction(lidar_id, no_scans_avg)
        else:
            print('Unsupported wind reconstruction method')


    def simulate_statistical_wind(self, lidar_id, scan_config, atmo_config, turb_parameter=None):
        """
        

        Parameters
        ----------
        lidar_id : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        for i in range(scan_config['no_10min_scans']):
            self.generate_VAD_scan(lidar_id,scan_config)
            self.generate_uncertainties(lidar_id)
            #atmo_config.pop('flow_model')
            self.generate_flow_field(lidar_id,atmo_config,atmo_config['flowfield_type'], turb_parameter)
            self.project_to_los(lidar_id)
            self.remove_samples(lidar_id, scan_config)
            self.reconstruct_wind(lidar_id, scan_config['WFR_algorithm'], scan_config['10min_average'])
            sonic_pos = np.array([[data.probing[lidar_id].lidar_pos_x.values,
                                  data.probing[lidar_id].lidar_pos_y.values,
                                  scan_config['range']]])
            time_steps = self.data.probing['hugin'].time.values
            self.generate_virtual_sonic(sonic_pos, time_steps)
            if i == 0:
                self.data.tenMin_wind = self.data.rc_wind.copy()
                self.data.sonic_wind_tenMin = self.data.sonic_wind.mean()
            else:
                new_idx = self.data.tenMin_wind.scan.max().values
                self.data.rc_wind['scan'] = self.data.rc_wind['scan'] + new_idx
                self.data.tenMin_wind = xr.concat(
                    [self.data.tenMin_wind, self.data.rc_wind], dim="scan")
                self.data.sonic_wind_tenMin = xr.concat(
                    [self.data.sonic_wind_tenMin, self.data.sonic_wind.mean()], dim='scan')


