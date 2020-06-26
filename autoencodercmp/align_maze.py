import numpy as np

def align_maze(idx_trials, activity_list, frame_trial, maze_position, reshape=False):
    'Align activity data by binning based on maze position.'

    # idx_trials: list of the trial indexes of the trials involved in activity_list
    # frame_trial: data['beh_dict']['frameTrialMem']
    # maze_position: data['beh_dict']['posF'][0,:]
    # reshape: each output of the RNN is 3D (n_trial=1, n_seq, n_feature), need to be reshaped to 2D
    # output: n_trial by n_neuron by n_seq

    # set bins
    min_posF = 10
    max_posF = 250
    n_bin = 20
    bin_spacing = 12
    bin_half_width = 6

    maze_position[np.isnan(maze_position)] = -100  # Set to a number that we can ignore

    # n_trails by n_neuron by n_bin
    shape = activity_list[0].shape[-1] #n_neuron (n_feature)
    activity_binned = np.zeros((len(idx_trials), shape, n_bin)) #shape: n_neuron by n_seq, for the convinence of plot

    for (i, idx_trial) in enumerate(idx_trials):
        maze_position_i = maze_position[frame_trial == idx_trial]
        activity_i = activity_list[i]
        if reshape == True:
            activity_i = activity_i.reshape(activity_i.shape[1], activity_i.shape[2])
        activity_binned[i, :, :] = np.array([np.mean(
            activity_i[np.logical_and(maze_position_i >= bin_center - bin_half_width,
                                      maze_position_i < bin_center + bin_half_width)], axis=0)
            for bin_center in np.arange(min_posF + bin_half_width,
                                        max_posF, bin_spacing)]).T

    return activity_binned