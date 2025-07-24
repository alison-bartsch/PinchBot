import math
import torch
import json
import numpy as np
import open3d as o3d 
from os.path import exists
from PIL import Image
from scipy.spatial.transform import Rotation

class ClayDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, pred_horizon, n_datapoints, n_raw_trajectories, center_action, rotate_goal=True):
        """
        The Dataloader for the clay sculpting dataset at the Trajectory level (compatible with ACT and Diffusion Policy). 

        :param episode_idxs: list of indices of the episodes to load
        :param dataset_dir: directory where the dataset is stored
        :param n_datapoints: number of datapoints (i.e. desired number of final trajectories after augmentation)
        :param n_raw_trajectories: number of raw trajectories in the dataset
        :param center_action: whether to center the action before normalizing
        """
        super(ClayDataset).__init__()
        self.dataset_dir = dataset_dir
        self.pred_horizon = pred_horizon
        self.n_datapoints = n_datapoints
        self.n_raw_trajectories = n_raw_trajectories
        self.center_action = center_action
        self.rotate_goal = rotate_goal
        self.ee_center = np.array([0.608, 0.014, 0.125])
        self.pcl_center = np.array([0.630, -0.0054, 0.074])

        self.a_mins7d = np.array([0.52776, -0.0662, 0.1272, -3.484, -10.10, -117, 0.008])
        self.a_maxs7d = np.array([0.68825, 0.09425, 0.1600, 49.3, 11.68, 207, 0.016])

        # determine the number of datapoints per trajectory - needs to be a round number
        self.n_datapoints_per_trajectory = self.n_datapoints / self.n_raw_trajectories
        if not self.n_datapoints_per_trajectory.is_integer():
            raise ValueError('The number of datapoints per trajectory needs to be a round number, please input a valid number of datapoints given the number of raw trajectories')

        # deterime the augmentation interval
        self.aug_step = 360 / self.n_datapoints_per_trajectory

    def get_dataset_min_max_stats(self):
        return self.a_mins7d, self.a_maxs7d

    def _center_pcl(self, pcl, center):
        centered_pcl = pcl - center
        centered_pcl = centered_pcl * 10
        return centered_pcl

    def _normalize_action(self, action):
        norm_action = (action - self.a_mins7d) / (self.a_maxs7d - self.a_mins7d)
        norm_action = norm_action  * 2 - 1 # set to [-1, 1]
        return norm_action
    
    def _rotate_pcl(self, state, center, rot):
        '''
        Faster implementation of rotation augmentation to fix slow down issue
        '''
        state = state - center
        R = Rotation.from_euler('xyz', np.array([0, 0, -rot]), degrees=True).as_matrix()
        state = R @ state.T
        pcl_aug = state.T + center
        return pcl_aug

    def _fix_real_action(self, action7d):
        if action7d[5] > 225:
            action7d[5] = -(360 - action7d[5])
        # check if in the unexecutable zone
        if action7d[5] < -117 and action7d[5] >= -135:
            action7d[5] = -117
        # check if need to wrap angles for unexecutable zone
        if action7d[5] < -120:
            action7d[5] = 180 + 180 - np.abs(action7d[5])
        # check if need to wrap angles for unexecutable zone
        if action7d[5] > 207:
            action7d[5] = 207

        if action7d[3] < -45:
            action7d[3] = 360 + action7d[3]
        elif action7d[3] > 45:
            action7d[3] = action7d[3] - 360
        return action7d

    def _rotate_action(self, action, center, rot):
        # given the center and rot about z in degrees, create the transform to for the points action[0:2]
        pts = np.array([[action[0], action[1], action[2]]])
        # rotate pts about center by rot degrees
        pts = pts - center
        R = Rotation.from_euler('z', np.radians(-rot), degrees=False).as_matrix()
        pts = R @ pts.T
        pts = pts.T + center
        x = pts[0, 0]
        y = pts[0, 1]

        # testing xyz convention
        R_obj_in_world = Rotation.from_euler('xyz', [action[3], action[4], action[5]], degrees=True)
        R_newframe_in_world = Rotation.from_euler('z', rot, degrees=True)
        R_obj_in_newframe = R_newframe_in_world.inv() * R_obj_in_world
        rx_new, ry_new, rz_new = R_obj_in_newframe.as_euler('xyz', degrees=True)

        action_aug = np.array([x, y, action[2], rx_new, ry_new, rz_new, action[6]]) # NOTE: for now we are keeping rx and ry the same

        # first check rz to wrap within expected range
        if action_aug[5] > 225:
            action_aug[5] = -(360 - action_aug[5])
        # check if in the unexecutable zone
        if action_aug[5] < -117 and action_aug[5] >= -135:
            action_aug[5] = -117
        # check if need to wrap angles for unexecutable zone
        if action_aug[5] < -120:
            action_aug[5] = 180 + 180 - np.abs(action_aug[5])
        # check if need to wrap angles for unexecutable zone
        if action_aug[5] > 207:
            action_aug[5] = 207
            
        return action_aug
    
    def __len__(self):
        """
        Return the number of episodes in the dataset (i.e. the number of actions in the trajectory folder)
        """
        return self.n_datapoints

    def __getitem__(self, idx):
        raw_traj_idx = int(idx // self.n_datapoints_per_trajectory) 
        # determine the rotation augmentation to apply
        aug_rot = (idx % self.n_datapoints_per_trajectory) * self.aug_step
        traj_path = self.dataset_dir + '/Trajectory' + str(raw_traj_idx)

        states = []
        actions = []
        j = 0

        while exists(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy'):  
            # ctr = np.load(traj_path + '/pcl_center' + str(j) + '.npy')
            s = np.load(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy')
            s_rot = self._rotate_pcl(s, self.pcl_center, aug_rot)
            s_rot_scaled = self._center_pcl(s_rot, self.pcl_center)
            states.append(s_rot_scaled)

            if j != 0:
                # load unnormalized action
                a = np.load(traj_path + '/action7d_unnormalized' + str(j-1) + '.npy')
                a = self._fix_real_action(a)
                a_rot = self._rotate_action(a, self.ee_center, aug_rot)

                a_rot = a # for now, do not rotate the action, just normalize it
                if self.center_action:
                    a_scaled = self._center_normalize_action(a_rot, self.ee_center)
                else:
                    a_scaled = self._normalize_action(a_rot)
                actions.append(a_scaled)
            j+=1

        episode_len = len(actions)
        start_ts = np.random.choice(episode_len)
        state = states[start_ts]
        
        # load uncentered goal
        g = np.load(traj_path + '/unnormalized_pointcloud' + str(j-1) + '.npy') # set the goal point cloud to be the last pcl in demo trajectory
        if self.rotate_goal:
            g_rot = self._rotate_pcl(g, self.pcl_center, aug_rot)
            goal = self._center_pcl(g_rot, self.pcl_center)
        
        else:
            goal = self._center_pcl(g, self.pcl_center)

        action = actions[start_ts:]
        action = np.stack(action, axis=0)

        # add in termination token -1 continue, 1 stop
        stop_token = -1 * np.ones((action.shape[0], 1))
        stop_token[-1] = 1
        action = np.concatenate((action, stop_token), axis=1)
        
        action_len = episode_len - start_ts

        if start_ts != 0:
            obs_pos = actions[start_ts-1]
        else:
            if self.center_action:
                obs_pos = self._center_normalize_action(np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04]), self.ee_center)
            else:
                obs_pos = self._normalize_action(np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04]))
        
        # add padding to obs_pos of one 0 vector to make 8d
        obs_pos = np.concatenate((obs_pos, -1 * np.ones((1))), axis=0)

        if action_len < self.pred_horizon:
            padded_action = np.zeros((self.pred_horizon, 8))
            padded_action[:action_len] = action
            for i in range(action_len, self.pred_horizon):
                padded_action[i] = action[-1]
        else:
            padded_action = action[:self.pred_horizon]

        # construct observations
        state_data = torch.from_numpy(state)
        goal_data = torch.from_numpy(goal).float()
        action_data = torch.from_numpy(padded_action).float()
        obs_pos_data = torch.from_numpy(obs_pos).float()

        nsample = dict()
        nsample['pointcloud'] = state_data
        nsample['goal'] = goal_data
        nsample['action'] = action_data
        nsample['agent_pos'] = obs_pos_data
        return nsample

class ClayDatasetContinualGuidance(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, pred_horizon, n_datapoints, n_raw_trajectories, center_action):
        '''
        '''
        super(ClayDatasetContinualGuidance).__init__()
        self.dataset_dir = dataset_dir
        self.pred_horizon = pred_horizon
        self.n_datapoints = n_datapoints
        self.n_raw_trajectories = n_raw_trajectories
        self.center_action = center_action
        self.ee_center = np.array([0.608, 0.014, 0.125])
        self.pcl_center = np.array([0.630, -0.0054, 0.074])

        self.a_mins7d = np.array([0.52776, -0.0662, 0.1272, -3.484, -10.10, -117, 0.008])
        self.a_maxs7d = np.array([0.68825, 0.09425, 0.1600, 49.3, 11.68, 207, 0.016])

        # determine the number of datapoints per trajectory - needs to be a round number
        self.n_datapoints_per_trajectory = self.n_datapoints / self.n_raw_trajectories
        if not self.n_datapoints_per_trajectory.is_integer():
            raise ValueError('The number of datapoints per trajectory needs to be a round number, please input a valid number of datapoints given the number of raw trajectories')

        # deterime the augmentation interval
        self.aug_step = 360 / self.n_datapoints_per_trajectory

    def get_dataset_min_max_stats(self):
        return self.a_mins7d, self.a_maxs7d

    def _center_pcl(self, pcl, center):
        centered_pcl = pcl - center
        centered_pcl = centered_pcl * 10
        return centered_pcl

    def _normalize_action(self, action):
        norm_action = (action - self.a_mins7d) / (self.a_maxs7d - self.a_mins7d)
        norm_action = norm_action  * 2 - 1 # set to [-1, 1]
        return norm_action
    
    def _rotate_pcl(self, state, center, rot):
        '''
        Faster implementation of rotation augmentation to fix slow down issue
        '''
        state = state - center
        R = Rotation.from_euler('xyz', np.array([0, 0, -rot]), degrees=True).as_matrix()
        state = R @ state.T
        pcl_aug = state.T + center
        return pcl_aug

    def _fix_real_action(self, action7d):
        if action7d[5] > 225:
            action7d[5] = -(360 - action7d[5])
        # check if in the unexecutable zone
        if action7d[5] < -117 and action7d[5] >= -135:
            action7d[5] = -117
        # check if need to wrap angles for unexecutable zone
        if action7d[5] < -120:
            action7d[5] = 180 + 180 - np.abs(action7d[5])
        # check if need to wrap angles for unexecutable zone
        if action7d[5] > 207:
            action7d[5] = 207

        if action7d[3] < -45:
            action7d[3] = 360 + action7d[3]
        elif action7d[3] > 45:
            action7d[3] = action7d[3] - 360
        return action7d

    def _rotate_action(self, action, center, rot):
        # given the center and rot about z in degrees, create the transform to for the points action[0:2]
        pts = np.array([[action[0], action[1], action[2]]])
        # rotate pts about center by rot degrees
        pts = pts - center
        R = Rotation.from_euler('z', np.radians(-rot), degrees=False).as_matrix()
        pts = R @ pts.T
        pts = pts.T + center
        x = pts[0, 0]
        y = pts[0, 1]


        # testing xyz convention
        R_obj_in_world = Rotation.from_euler('xyz', [action[3], action[4], action[5]], degrees=True)
        R_newframe_in_world = Rotation.from_euler('z', rot, degrees=True)
        R_obj_in_newframe = R_newframe_in_world.inv() * R_obj_in_world
        rx_new, ry_new, rz_new = R_obj_in_newframe.as_euler('xyz', degrees=True)

        action_aug = np.array([x, y, action[2], rx_new, ry_new, rz_new, action[6]]) # NOTE: for now we are keeping rx and ry the same

        # first check rz to wrap within expected range
        if action_aug[5] > 225:
            action_aug[5] = -(360 - action_aug[5])
        # check if in the unexecutable zone
        if action_aug[5] < -117 and action_aug[5] >= -135:
            action_aug[5] = -117
        # check if need to wrap angles for unexecutable zone
        if action_aug[5] < -120:
            action_aug[5] = 180 + 180 - np.abs(action_aug[5])
        # check if need to wrap angles for unexecutable zone
        if action_aug[5] > 207:
            action_aug[5] = 207
            
        return action_aug
    
    def __len__(self):
        """
        Return the number of episodes in the dataset (i.e. the number of actions in the trajectory folder)
        """
        return self.n_datapoints

    def __getitem__(self, idx):
        raw_traj_idx = int(idx // self.n_datapoints_per_trajectory) 
        # determine the rotation augmentation to apply
        aug_rot = (idx % self.n_datapoints_per_trajectory) * self.aug_step
        traj_path = self.dataset_dir + '/Trajectory' + str(raw_traj_idx)

        states = []
        actions = []
        j = 0

        while exists(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy'):  
            s = np.load(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy')
            s_rot = self._rotate_pcl(s, self.pcl_center, aug_rot)
            s_rot_scaled = self._center_pcl(s_rot, self.pcl_center)
            states.append(s_rot_scaled)

            if j != 0:
                # load unnormalized action
                a = np.load(traj_path + '/action7d_unnormalized' + str(j-1) + '.npy')
                a = self._fix_real_action(a)
                a_rot = self._rotate_action(a, self.ee_center, aug_rot)

                a_rot = a # for now, do not rotate the action, just normalize it
                if self.center_action:
                    a_scaled = self._center_normalize_action(a_rot, self.ee_center)
                else:
                    a_scaled = self._normalize_action(a_rot)
                actions.append(a_scaled)
            j+=1

        episode_len = len(actions)
        start_ts = np.random.choice(episode_len)
        state = states[start_ts]

        # create an array going linearly from -1 to 1 with the length of the episode
        guidance = np.linspace(-1, 1, episode_len)
        # expand dimensions
        guidance = np.expand_dims(guidance, axis=1)
        
        # load uncentered goal
        g = np.load(traj_path + '/unnormalized_pointcloud' + str(j-1) + '.npy') # set the goal point cloud to be the last pcl in demo trajectory
        g_rot = self._rotate_pcl(g, self.pcl_center, aug_rot)
        goal = self._center_pcl(g_rot, self.pcl_center)

        action = actions[start_ts:]
        action = np.stack(action, axis=0)
        action = np.concatenate((action, guidance[start_ts:]), axis=1)
        
        action_len = episode_len - start_ts

        if start_ts != 0:
            obs_pos = actions[start_ts-1]
        else:
            if self.center_action:
                obs_pos = self._center_normalize_action(np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04]), self.ee_center)
            else:
                obs_pos = self._normalize_action(np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04]))
        
        # add padding to obs_pos of one 0 vector to make 8d
        obs_pos = np.concatenate((obs_pos, -1 * np.ones((1))), axis=0)

        if action_len < self.pred_horizon:
            padded_action = np.zeros((self.pred_horizon, 8))
            padded_action[:action_len] = action
            for i in range(action_len, self.pred_horizon):
                padded_action[i] = action[-1]
        else:
            padded_action = action[:self.pred_horizon]

        # construct observations
        state_data = torch.from_numpy(state)
        goal_data = torch.from_numpy(goal).float()
        action_data = torch.from_numpy(padded_action).float()
        obs_pos_data = torch.from_numpy(obs_pos).float()

        nsample = dict()
        nsample['pointcloud'] = state_data
        nsample['goal'] = goal_data
        nsample['action'] = action_data
        nsample['agent_pos'] = obs_pos_data
        return nsample
    
class SubGoalClayDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, pred_horizon, n_datapoints, n_raw_trajectories, center_action, subgoal_stepsize):
        """
        The Dataloader for the clay sculpting dataset at the Trajectory level (compatible with ACT and Diffusion Policy). 

        :param episode_idxs: list of indices of the episodes to load
        :param dataset_dir: directory where the dataset is stored
        :param n_datapoints: number of datapoints (i.e. desired number of final trajectories after augmentation)
        :param n_raw_trajectories: number of raw trajectories in the dataset
        :param center_action: whether to center the action before normalizing
        """
        super(SubGoalClayDataset).__init__()
        self.dataset_dir = dataset_dir
        self.pred_horizon = pred_horizon
        self.n_datapoints = n_datapoints
        self.n_raw_trajectories = n_raw_trajectories
        self.center_action = center_action
        self.subgoal_stepsize = subgoal_stepsize

        self.ee_center = np.array([0.608, 0.014, 0.125])
        self.pcl_center = np.array([0.630, -0.0054, 0.074])

        self.a_mins7d = np.array([0.52776, -0.0662, 0.1272, -3.484, -10.10, -117, 0.008])
        self.a_maxs7d = np.array([0.68825, 0.09425, 0.1600, 49.3, 11.68, 207, 0.016])

        # determine the number of datapoints per trajectory - needs to be a round number
        self.n_datapoints_per_trajectory = self.n_datapoints / self.n_raw_trajectories
        if not self.n_datapoints_per_trajectory.is_integer():
            raise ValueError('The number of datapoints per trajectory needs to be a round number, please input a valid number of datapoints given the number of raw trajectories')

        # deterime the augmentation interval
        self.aug_step = 360 / self.n_datapoints_per_trajectory
        
    def get_dataset_min_max_stats(self):
        return self.a_mins7d, self.a_maxs7d

    def _center_pcl(self, pcl, center):
        centered_pcl = pcl - center
        centered_pcl = centered_pcl * 10
        return centered_pcl

    def _normalize_action(self, action):
        norm_action = (action - self.a_mins7d) / (self.a_maxs7d - self.a_mins7d)
        norm_action = norm_action  * 2 - 1 # set to [-1, 1]
        return norm_action
    
    def _rotate_pcl(self, state, center, rot):
        '''
        Faster implementation of rotation augmentation to fix slow down issue
        '''
        state = state - center
        R = Rotation.from_euler('xyz', np.array([0, 0, -rot]), degrees=True).as_matrix()
        state = R @ state.T
        pcl_aug = state.T + center
        return pcl_aug

    def _rotate_action(self, action, center, rot):
        # given the center and rot about z in degrees, create the transform to for the points action[0:2]
        pts = np.array([[action[0], action[1], action[2]]])
        # rotate pts about center by rot degrees
        pts = pts - center
        R = Rotation.from_euler('z', np.radians(-rot), degrees=False).as_matrix()
        pts = R @ pts.T
        pts = pts.T + center
        x = pts[0, 0]
        y = pts[0, 1]

        # testing xyz convention
        R_obj_in_world = Rotation.from_euler('xyz', [action[3], action[4], action[5]], degrees=True)
        R_newframe_in_world = Rotation.from_euler('z', rot, degrees=True)
        R_obj_in_newframe = R_newframe_in_world.inv() * R_obj_in_world
        rx_new, ry_new, rz_new = R_obj_in_newframe.as_euler('xyz', degrees=True)

        action_aug = np.array([x, y, action[2], rx_new, ry_new, rz_new, action[6]]) # NOTE: for now we are keeping rx and ry the same

        # first check rz to wrap within expected range
        if action_aug[5] > 225:
            action_aug[5] = -(360 - action_aug[5])
        # check if in the unexecutable zone
        if action_aug[5] < -117 and action_aug[5] >= -135:
            action_aug[5] = -117
        # check if need to wrap angles for unexecutable zone
        if action_aug[5] < -120:
            action_aug[5] = 180 + 180 - np.abs(action_aug[5])
        # check if need to wrap angles for unexecutable zone
        if action_aug[5] > 207:
            action_aug[5] = 207
        return action_aug
    
    def _wrap_rz(self, original_rz):
        wrapped_rz = (original_rz + 90) % 180 - 90
        return wrapped_rz

    def _fix_real_action(self, action7d):
        if action7d[5] > 225:
            action7d[5] = -(360 - action7d[5])
        # check if in the unexecutable zone
        if action7d[5] < -117 and action7d[5] >= -135:
            action7d[5] = -117
        # check if need to wrap angles for unexecutable zone
        if action7d[5] < -120:
            action7d[5] = 180 + 180 - np.abs(action7d[5])
        # check if need to wrap angles for unexecutable zone
        if action7d[5] > 207:
            action7d[5] = 207

        if action7d[3] < -45:
            action7d[3] = 360 + action7d[3]
        elif action7d[3] > 45:
            action7d[3] = action7d[3] - 360
        return action7d
    
    def __len__(self):
        """
        Return the number of episodes in the dataset (i.e. the number of actions in the trajectory folder)
        """
        return self.n_datapoints

    def __getitem__(self, idx):
        raw_traj_idx = int(idx // self.n_datapoints_per_trajectory) 
        # determine the rotation augmentation to apply
        aug_rot = (idx % self.n_datapoints_per_trajectory) * self.aug_step
        traj_path = self.dataset_dir + '/Trajectory' + str(raw_traj_idx)

        states = []
        actions = []
        j = 0

        while exists(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy'):  
            s = np.load(traj_path + '/unnormalized_pointcloud' + str(j) + '.npy')
            s_rot = self._rotate_pcl(s, self.pcl_center, aug_rot)
            s_rot_scaled = self._center_pcl(s_rot, self.pcl_center)
            states.append(s_rot_scaled)

            if j != 0:
                # load unnormalized action
                a = np.load(traj_path + '/action7d_unnormalized' + str(j-1) + '.npy')
                a = self._fix_real_action(a)
                a_rot = self._rotate_action(a, self.ee_center, aug_rot)
                if self.center_action:
                    a_scaled = self._center_normalize_action(a_rot, self.ee_center)
                else:
                    a_scaled = self._normalize_action(a_rot)
                actions.append(a_scaled)
            j+=1

        full_episode_len = len(actions)
        start_ts = np.random.choice(full_episode_len - self.subgoal_stepsize - 1)
        full_action_len = full_episode_len - start_ts

        if full_action_len >= self.pred_horizon:
            action = actions[start_ts:start_ts + self.pred_horizon]
            state_list = states[start_ts:(start_ts + self.pred_horizon + self.subgoal_stepsize):self.subgoal_stepsize]
        else:
            action = actions[start_ts:]
            state_list = states[start_ts::self.subgoal_stepsize]

        action_len = len(action)
        action = np.stack(action, axis=0)

        # add in termination token -1 continue, 1 stop
        stop_token = -1 * np.ones((action.shape[0], 1))
        stop_token[-1] = 1
        action = np.concatenate((action, stop_token), axis=1)
        
        if start_ts != 0:
            obs_pos = actions[start_ts-1]
        else:
            if self.center_action:
                obs_pos = self._center_normalize_action(np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04]), self.ee_center)
            else:
                obs_pos = self._normalize_action(np.array([0.6, 0.0, 0.165, 0.0, 0.0, 0.0, 0.04]))
        
        # add padding to obs_pos of one 0 vector to make 8d
        obs_pos = np.concatenate((obs_pos, -1 * np.ones((1))), axis=0)

        states_seq_size = int((self.pred_horizon + self.subgoal_stepsize) / self.subgoal_stepsize)

        if action_len < self.pred_horizon:
            padded_action = np.zeros((self.pred_horizon, 8))
            padded_action[:action_len] = action
            for i in range(action_len, self.pred_horizon):
                padded_action[i] = action[-1]

            padded_states = np.zeros((states_seq_size, 2048, 3))
            padded_states[:len(state_list)] = np.stack(state_list, axis=0)
            padded_states[len(state_list):] = np.tile(state_list[-1], (len(padded_states[len(state_list):]), 1, 1))
        else:
            padded_action = action[:self.pred_horizon]
            padded_states = np.stack(state_list, axis=0)

        # construct observations
        padded_states_data = torch.from_numpy(padded_states).float()
        action_data = torch.from_numpy(padded_action).float()
        obs_pos_data = torch.from_numpy(obs_pos).float()

        nsample = dict()
        nsample['pcl_seq'] = padded_states_data
        nsample['action'] = action_data
        nsample['agent_pos'] = obs_pos_data
        return nsample