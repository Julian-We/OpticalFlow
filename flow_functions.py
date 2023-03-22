import os
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tfle
from tqdm import tqdm
import pandas as pd
from skimage.registration import optical_flow_tvl1 as of  # as optical_flow rename to optical flow later
from skimage import segmentation, filters
from scipy.signal import correlate
from scipy import ndimage as ndi
import matplotlib.animation as animation
from scipy.stats import pearsonr as pcorelate
import cv2

def get_value(value1, value2, direction):
    if value1 > value2:
        return value1 if direction == 'higher' else value2
    elif value1 < value2:
        return value2 if direction == 'higher' else value1
    elif value1 == value2:
        return value1


def drift_correct_crop(root: str,
                       img_pth: str):
    """
    For now neighter a drift correction nor a crop function ^^'
    Loads image using tifffile. Flips axes 0 and 1 (for Fiji images this goes from "TCYX" to "CTYX")


    :param root: Path in which images to analyse are located
    :param img_pth: name of the image that should be preprocessed
    :return: image as numpy array and the order of the axes
    """
    img = tfle.imread(os.path.join(root, img_pth))  # Read image file

    img = np.transpose(img, axes=[1, 0, 2, 3])  # Swap channel and time axes (allows for easier code later)
    # print(img.shape)
    axes = 'CTYX'
    return img, axes


def plot_flow(img, flow_data, timepoints):
    image0, image1 = img[timepoints]

    u, v = flow_data[0:2]
    # --- Compute flow magnitude
    norm = np.sqrt(u ** 2 + v ** 2)

    # --- Display
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6))

    # --- Sequence image sample

    ax0.imshow(image0, cmap='gray')
    ax0.set_title("Reference image")
    ax0.axhline(225, alpha=0.5, color='red')

    ax0.set_axis_off()

    ax1.imshow(image1, cmap='gray')
    ax1.set_title("Moved image")
    ax1.axhline(225, alpha=0.5, color='red')
    ax1.set_axis_off()

    # --- Quiver plot arguments

    nvec = 20  # Number of vectors to be displayed along each image dimension
    nl, nc = image0.shape
    step = max(nl // nvec, nc // nvec)

    y, x = np.mgrid[:nl:step, :nc:step]
    u_ = u[::step, ::step]
    v_ = v[::step, ::step]

    ax2.imshow(image0, cmap='gray')
    ax2.quiver(x, y, v_, u_, color='r', units='dots',
               angles='xy', scale_units='xy', lw=3)
    ax2.set_title("Optical flow magnitude and vector field")
    ax2.set_axis_off()
    fig.tight_layout()

    extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig('ax2_figure.png', bbox_inches=extent)

    plt.show()


def get_image_flow(img, timepoints, plot=False):
    """

    :param img: Single channel image
    :param timepoints: Array or tuple of two integers
    :param plot: should the result be plotted
    :return:
    """
    img_ref = img[timepoints[0]]
    img_mov = img[timepoints[1]]

    y_flow, x_flow = of(img_ref, img_mov)
    nrm = np.sqrt(y_flow ** 2 + x_flow ** 2)

    yx_flow = np.stack((y_flow, x_flow), axis=-1)

    if plot:
        plot_flow(img, [y_flow, x_flow, nrm, yx_flow], timepoints)
    return y_flow, x_flow, nrm, yx_flow


def sth(flow):
    envec = 20
    y_ax, x_ax = flow[2].shape
    steps = max(y_ax // envec, x_ax // envec)
    h, j = flow[0], flow[1]

    yy, xx = np.mgrid[:y_ax:steps, :x_ax:steps]

    h_ = h[::steps, ::steps]
    j_ = j[::steps, ::steps]

    return yy, xx, h_, j_


def angle_between(v1, v2):
    # LEGACY CONTENT
    # dot_pr = v1.dot(v2)
    # norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    #
    # return np.rad2deg(np.arccos(dot_pr / norms))
    dot_pr = v1.dot(v2)
    det = np.linalg.det([v1, v2])

    return -np.rad2deg(np.arctan2(det, dot_pr))


def correlations(mode: str, image_g, image_n, size=8, **kwargs):
    """
    Performs correlations on the nucleus data and the

    :param mode: can be either "pearson" or "xcorr". If neither is given it tries to guess or defaults to 'pearson'
    :return:
    """

    if mode.lower() not in ['pearson', 'xcorr']:
        try:
            mode = 'pearson' if len(image_n.shape) == 3 else 'xcorr'
        except Exception as exept:
            print(exept)
            mode = 'pearson'

    if mode.lower() == 'pearson':
        y, x, _ = image_n.shape  #, v
    else:
        y, x = image_n.shape  #, v
    y_fitn = y // size
    x_fitn = x // size
    y_start = int((y - (y_fitn * size)) / 2)
    x_start = int((x - (x_fitn * size)) / 2)
    wind_multi = 0.59375 if 'wind_multi' not in kwargs.keys() else kwargs['wind_multi']  # Default is 0.59375
    win_size = int(wind_multi * size)

    # Empty Matrix for output of the pearson correlation
    pc_out = np.zeros((y_fitn, x_fitn))

    # Empty ouput arrays for cross correlation
    vec_mtx_out = np.zeros((y_fitn, x_fitn, 2))
    dp_out = np.zeros((y_fitn, x_fitn))

    mids = []
    mids_soll = []
    corrs = []

    for idxy, y_dim in enumerate(range(y_fitn)):
        for idxx, x_dim in enumerate(range(x_fitn)):

            low_y = y_start + (idxy * size)
            mid_y = low_y + size / 2
            high_y = y_start + (idxy * size) + size

            low_sw_y = int(mid_y - win_size) if (mid_y - win_size) > 0 else 0
            high_sw_y = int(mid_y + win_size) if (mid_y + win_size) < y else y

            low_x = x_start + (idxx * size)
            mid_x = low_x + size / 2
            high_x = x_start + (idxx * size) + size

            low_sw_x = int(mid_x - win_size) if (mid_x - win_size) > 0 else 0
            high_sw_x = int(mid_x + win_size) if (mid_x + win_size) < x else x
            if mode.lower() == 'pearson':
                snip_n = image_n[low_y:high_y, low_x:high_x].flatten()
                snip_g = image_g[low_y:high_y, low_x:high_x].flatten()

                corr_value, p_value = pcorelate(snip_g - snip_g.mean(), snip_n - snip_n.mean())

                pc_out[idxy, idxx] = corr_value
            if mode.lower() == 'xcorr':
                snip = image_g[low_y:high_y, low_x:high_x]

                snip_search = image_n[low_sw_y:high_sw_y, low_sw_x:high_sw_x]
                p_corr = correlate(snip_search,  # - snip_search.mean(),
                                   snip,  # - snip.mean(),
                                   method='fft')
                p_middle = np.asarray(np.unravel_index(p_corr.argmax(), p_corr.shape))
                # if len(p_middle.shape) > 3:
                p_middle = p_middle[:2]
                mids.append(p_middle)
                mid_sw = np.asarray(snip_search.shape)
                mids_soll.append(mid_sw.astype(int))
                corrs.append(p_corr)
                disp_vec = [p_middle[0] - mid_sw[0], p_middle[1] - mid_sw[1]]
                dips_prod = np.sqrt(disp_vec[0] ** 2 + disp_vec[1] ** 2)
                vec_mtx_out[idxy, idxx] = disp_vec
                dp_out[idxy, idxx] = dips_prod

    return [pc_out] if mode.lower() == 'pearson' else [vec_mtx_out, dp_out]


def cbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def plot_all_ani(idx, *fargs):
    movie, info, frame = fargs
    frame = frame * idx

    bbg = np.zeros(info['shape yx'])  # Create black background for directional analysis

    # print(idx)
    # fig = plt.figure(figsize=(30, 30))
    # ax1 = fig.add_subplot(3,3,1)
    # ax2 = fig.add_subplot(3,3,2)
    # ax3 = fig.add_subplot(3,3,3)
    # ax4 = fig.add_subplot(3,3,4)
    # ax5 = fig.add_subplot(3,3,5)
    # ax6 = fig.add_subplot(3,3,6)
    # ax7 = fig.add_subplot(3,3,7)
    # ax8 = fig.add_subplot(3,3,(8, 9))
    # print(f'flow_g shape: {info["flow_n"].shape}\n',f'flow_g[0] shape: {info["flow_g"][0].shape}\n')

    ax1.imshow(bbg, cmap='gray')
    ax1.imshow(movie[info['ch_nucleus'], frame], cmap='Blues_r',
               alpha=movie[info['ch_nucleus'], frame]/movie[info['ch_nucleus'], frame].max())
    ax1.imshow(movie[info['ch_granules'], frame], cmap='Reds_r',
               alpha=movie[info['ch_granules'], frame]/movie[info['ch_granules'], frame].max())

    yy, xx, nh_, nj_ = sth(info['flow_n'][idx])
    _, _, gh_, gj_ = sth(info['flow_g'][idx])

    ax2.clear()
    ax2.imshow(info['flow_n'][idx][2], vmin=info['flow_minmax'][0], vmax=info['flow_minmax'][1])
    ax2.quiver(xx, yy, nj_, nh_, units='dots',
               angles='xy', scale_units='xy', lw=3)
    ax2.set_title('Nucleus flow')

    ax3.clear()
    ax3.imshow(info['flow_g'][idx][2], vmin=info['flow_minmax'][0], vmax=info['flow_minmax'][1])
    ax3.quiver(xx, yy, gj_, gh_, units='dots',
               angles='xy', scale_units='xy', lw=3)
    ax3.set_title('Granule flow')

    ax4.clear()
    a_range = 180
    flow_n_norm = info['flow_n'][idx][2] / info['flow_n'][idx][2].max()
    ax4.imshow(bbg, cmap='gray')
    ax4.imshow(info['angle_n'][idx], cmap='hsv', vmin=-a_range, vmax=a_range, alpha=flow_n_norm)
    ax4.set_title('Directionality of nucleus movement')
    ax4.axis('off')

    ax5.clear()
    flow_g_norm = info['flow_g'][idx][2] / info['flow_g'][idx][2].max()
    ax5.imshow(bbg, cmap='gray')
    ax5.imshow(info['angle_g'][idx], cmap='hsv', vmin=-a_range, vmax=a_range, alpha=flow_g_norm)
    ax5.set_title('Directionality of granule movement')
    ax5.axis('off')

    ax6.clear()
    ax6.hist(np.deg2rad(info['angle_n'][idx].flatten()), weights=info['flow_n'][idx][2].flatten(), bins=20,
             label='nucleus angles', alpha=0.5)
    ax6.hist(np.deg2rad(info['angle_g'][idx].flatten()), weights=info['flow_g'][idx][2].flatten(), bins=20,
             color='orange', label='granule angles', alpha=0.5)
    ax6.set_theta_zero_location("N")  # Set polar plot 0 to North
    # ax6.set_theta_direction(-1) # Mirror the polar plot
    ax6.set_xlabel(r'Diverson from the y-axis (direction "up") [in $\degree$]')
    ax6.legend()

    smooth = filters.gaussian(movie[info['ch_granules'], frame], sigma=4.5)
    thresh = smooth > filters.threshold_otsu(smooth)
    fill = ndi.binary_fill_holes(thresh)
    nucleus_seg = segmentation.clear_border(fill)


    ax7.clear()
    # ax7.imshow(bbg, cmap='gray')
    ax7.imshow(cv2.resize(info['pcorr'][idx], dsize=info['shape yx'], interpolation=cv2.INTER_NEAREST),
               vmin=-1, vmax=1, cmap='cool') #, alpha=movie[info['ch_granules'], frame]/movie[info['ch_granules'], frame].max()
    ax7.contour(nucleus_seg, colors='yellow', linestyles='-', linewidths=4)
    ax7.set_title('Pearson correlation of nucleus and granule flow vectors ')
    ax7.axis('off')

    # print(len(info['pcorr']))
    # print(list(range(len(info['pcorr_mean'][:idx]))), '\n',   info['pcorr_mean'][:idx])
    ax8.clear()
    ax8.plot(range(len(info['pcorr_mean'][:idx])), info['pcorr_mean'][:idx], label='Mean p.-correlation value')
    ax8.set_ylim(ymin=0.75, ymax=1.05)
    ax8.set_xlim(xmax=len(info['pcorr_mean']))


def plot_all(movie, info, para):
    idx, frame = para
    bbg = np.zeros([1024, 1024])  # Create black background for directional analysis
    fig = plt.figure(figsize=(30, 30))
    ax1 = fig.add_subplot(3, 3, 1)
    ax2 = fig.add_subplot(3, 3, 2)
    ax3 = fig.add_subplot(3, 3, 3)
    ax4 = fig.add_subplot(3, 3, 4)
    ax5 = fig.add_subplot(3, 3, 5)
    ax6 = fig.add_subplot(3, 3, 6, projection='polar')
    ax7 = fig.add_subplot(3, 3, 7)
    ax8 = fig.add_subplot(3, 3, (8, 9))

    # print(f'flow_g shape: {info["flow_n"].shape}\n',f'flow_g[0] shape: {info["flow_g"][0].shape}\n')
    ax1.imshow(bbg, cmap='gray')
    ax1.imshow(movie[info['ch_nucleus'], frame], cmap='Blues_r', alpha=movie[info['ch_nucleus'], frame]/movie[info['ch_nucleus'], frame].max())
    ax1.imshow(movie[info['ch_granules'], frame], cmap='Reds_r', alpha=movie[info['ch_granules'], frame]/movie[info['ch_granules'], frame].max())

    yy, xx, nh_, nj_ = sth(info['flow_n'][idx])
    _, _, gh_, gj_ = sth(info['flow_g'][idx])

    ax2.imshow(info['flow_n'][idx][2], vmin=info['flow_minmax'][0], vmax=info['flow_minmax'][1])
    ax2.quiver(xx, yy, nj_, nh_, units='dots',
               angles='xy', scale_units='xy', lw=3)
    ax2.set_title('Nucleus flow')

    ax3.imshow(info['flow_g'][idx][2], vmin=info['flow_minmax'][0], vmax=info['flow_minmax'][1])
    ax3.quiver(xx, yy, gj_, gh_, units='dots',
               angles='xy', scale_units='xy', lw=3)
    ax3.set_title('Granule flow')

    a_range = 180
    flow_n_norm = info['flow_n'][idx][2] / info['flow_n'][idx][2].max()
    ax4.imshow(bbg, cmap='gray')
    ax4.imshow(info['angle_n'][idx], cmap='hsv', vmin=-a_range, vmax=a_range, alpha=flow_n_norm)
    ax4.set_title('Directionality of nucleus movement')
    ax4.axis('off')

    flow_g_norm = info['flow_g'][idx][2] / info['flow_g'][idx][2].max()
    ax5.imshow(bbg, cmap='gray')
    ax5.imshow(info['angle_g'][idx], cmap='hsv', vmin=-a_range, vmax=a_range, alpha=flow_g_norm)
    ax5.set_title('Directionality of granule movement')
    ax5.axis('off')

    ax6.hist(np.deg2rad(info['angle_n'][idx].flatten()), bins=20, label='nucleus angles', alpha=0.5)
    ax6.hist(np.deg2rad(info['angle_g'][idx].flatten()), bins=20, color='orange', label='granule angles', alpha=0.5)
    # ax6.set_xlim(xmin=-50, xmax=50)
    # ax6.set_ylim(ymin=0, ymax=12000)
    # ax6.set_aspect(0.0085)
    ax6.set_theta_zero_location("N")
    # ax6.set_theta_direction(-1)
    ax6.set_xlabel(r'Diverson from the y-axis (direction "up") [in $\degree$]')
    ax6.legend()


    smooth = filters.gaussian(movie[info['ch_granules'], frame], sigma=4)
    thresh = smooth > filters.threshold_otsu(smooth)
    fill = ndi.binary_fill_holes(thresh)
    nucleus_seg = segmentation.clear_border(fill)

    # ax7.imshow(bbg, cmap='gray')
    ax7.imshow(cv2.resize(info['pcorr'][idx], dsize=info['shape yx'], interpolation=cv2.INTER_NEAREST),
               vmin=-1, vmax=1, cmap='cool') #, alpha=movie[info['ch_granules'], frame]/movie[info['ch_granules'], frame].max()
    ax7.contour(nucleus_seg, colors='yellow', linestyles='-', linewidths=4)
    ax7.set_title('Pearson correlation of nucleus and granule flow vectors ')
    ax7.axis('off')


    # print(len(info['pcorr']))
    # print(list(range(len(info['pcorr_mean'][:idx]))), '\n',   info['pcorr_mean'][:idx])
    ax8.plot(range(len(info['pcorr_mean'][:idx])), info['pcorr_mean'][:idx], label='Mean p.-correlation value')
    # ax8.set_ylim(ymin=0, ymax=1)
    ax8.set_xlim(xmax=len(info['pcorr_mean']))


def process_movie(mov,
                  axes: str,
                  animated=False,
                  **kwargs):
    """
    Iterates through a image series and calculates optical flow
    :param mov: image series as numpy array
    :param axes: sting of the order of axes; For example 'TCYX'
    :param animated: bool (default: False) | If True, animation of the image is made
    :param kwargs: {save_path: str, Export-path} {int_num: int, Sample every __ frame}
    {ch_nucleus : int, channel of nucleus} {ch_granules : int, channel of nucleus} {}
    """
    #Establish a save path if wanted, otherwise save results in the home folder
    save_path = '' if 'save_path' not in kwargs.keys() else kwargs['save_path']

    # Integer. each step is taken after the according
    int_num = 2 if 'int_num' not in kwargs.keys() else kwargs['int_num']

    if 'axes' not in kwargs.keys() and axes == 'CTYX':
        c, t, y, x = mov.shape
    else:
        if kwargs["axes"] != "CTYX":
            raise Warning(
                f'You entered axes in the shape {kwargs["axes"]} yet "CTYX" was expected. By default "TCYX" is used. You may experience problesms with this script ')
        t, c, y, x = kwargs["axes"]

    ch_nucleus = 2 if 'ch_nucleus' not in kwargs.keys() else kwargs['ch_nucleus']
    ch_granule = 1 if 'ch_granules' not in kwargs.keys() else kwargs['ch_granules']

    interval = t // int_num

    slices = [0, int_num]
    angle_data = {}
    flow_data = {}
    pcorr_data = {}
    xcorr_data = {}
    flow_g_max, flow_g_min = 0, 0
    flow_n_max, flow_n_min = 0, 0
    # plt.imshow(mov[ch_nucleus][0])
    for i, _ in tqdm(enumerate(range(interval - 1)), total=interval - 1, desc=f'Processed Frames'):
        flow_n = get_image_flow(mov[ch_nucleus], slices, plot=False)
        flow_g = get_image_flow(mov[ch_granule], slices, plot=False)
        flow_data.update({i: [flow_n, flow_g]})

        flow_g_max = get_value(flow_g_max, flow_g[2].max(), 'higher')
        flow_g_min = get_value(flow_g_min, flow_g[2].min(), 'lower')

        flow_n_max = get_value(flow_n_max, flow_n[2].max(), 'higher')
        flow_n_min = get_value(flow_n_min, flow_n[2].min(), 'lower')

        ref_x_axis = np.array([0, 1])
        angle_n = np.apply_along_axis(angle_between, 2, flow_n[3], ref_x_axis) - 90
        angle_g = np.apply_along_axis(angle_between, 2, flow_g[3], ref_x_axis) - 90
        angle_data.update({i: [angle_n, angle_g]})

        pcorr = correlations('pearson', flow_g[3], flow_n[3], size=4)
        pcorr_data.update({i: pcorr})

        xcorr = correlations('xcorr', angle_g, angle_n, size=16)
        xcorr_data.update({i: xcorr})

        slices = [x + int_num for x in slices]
        # if i == 10:
        #     break

    df_flow = pd.DataFrame.from_dict(flow_data, orient='index', columns=['flow_n', 'flow_g'])
    df_angle = pd.DataFrame.from_dict(angle_data, orient='index', columns=['angle_n', 'angle_g'])
    df_pcorr = pd.DataFrame.from_dict(pcorr_data, orient='index', columns=['pearson_corr'])
    df_xcorr = pd.DataFrame.from_dict(xcorr_data, orient='index', columns=['x_corr_vec', 'x_corr_dp'])
    df_master = pd.concat([df_flow, df_angle, df_pcorr, df_xcorr], axis=1)

    flow_min = get_value(flow_n_min, flow_g_min, 'higher')
    flow_max = get_value(flow_n_max, flow_g_max, 'higher')

    plot_info = {
        'ch_granules': ch_granule,
        'ch_nucleus': ch_nucleus,
        'shape yx': [y, x],
        'flow_n': df_master['flow_n'],
        'flow_g': df_master['flow_g'],
        'angle_n': df_master['angle_n'],
        'angle_g': df_master['angle_g'],
        'pcorr': df_master['pearson_corr'],
        'pcorr_mean': [p.mean() for p in df_master['pearson_corr']],
        'xcorr': [df_master['x_corr_vec'], df_master['x_corr_dp']],
        'flow_minmax': [flow_min, flow_max]
    }
    # print(len(plot_info['flow_n']))
    plot_all(mov, plot_info, [len(plot_info['pcorr_mean']) - 1, slices[0]])
    from random import randrange
    id = hex(randrange(1000000000000000))
    plt.savefig(os.path.join(save_path, f'figure_{id}.pdf'))
    # plt.close(fig)
    if animated:
        global fig, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8
        fig = plt.figure(figsize=(30, 30))
        ax1 = fig.add_subplot(3, 3, 1)
        ax2 = fig.add_subplot(3, 3, 2)
        ax3 = fig.add_subplot(3, 3, 3)
        ax4 = fig.add_subplot(3, 3, 4)
        ax5 = fig.add_subplot(3, 3, 5)
        ax6 = fig.add_subplot(3, 3, 6, projection='polar')
        ax7 = fig.add_subplot(3, 3, 7)
        ax8 = fig.add_subplot(3, 3, (8, 9))
        ani = animation.FuncAnimation(fig, plot_all_ani, fargs=[mov, plot_info, int_num], interval=1000 / 5,
                                      save_count=len(plot_info['pcorr_mean']) - 1)
        try:
            giff = animation.HTMLWriter(fps=5, bitrate=64000) #FFMpegWriter
        except:
            print('Something went wrong when creating the movie writer')
            giff = animation.PillowWriter(fps=5)
        # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ani.save(os.path.join(save_path, f'animation_{id}.html'), writer=giff)
        plt.close(fig)

