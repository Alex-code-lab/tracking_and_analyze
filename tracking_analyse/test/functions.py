def read_hdf5_all(pathway_experiment, condition, name_file='filtered_ok',
                  nbr_frame_min=200, drift=False, search_range: int = 100,
                  memory: int = 15, adaptive_stop: int = 30, min_mass: int = 1000, max_size: int = 40,):
    """
    Read all HDF5 trajectory files from a given experiment condition, and optionally apply drift correction
    and other preprocessing steps.

    Parameters:
    - pathway_experiment : list of str
        List of directory paths containing HDF5 trajectory files.
    - condition : str
        Description of the experimental condition to label the data.
    - name_file : str, optional
        Specific string to identify the correct HDF5 files (default is 'filtered_ok').
    - nbr_frame_min : int, optional
        Minimum number of frames a particle must be present to be included (default is 200).
    - drift : bool, optional
        Flag to apply drift correction to the trajectory data (default is False).
    - search_range, memory, adaptive_stop, min_mass, max_size : int, optional
        Parameters used in particle linking and filtering.

    Returns:
    - data_all : pd.DataFrame
        DataFrame containing concatenated and processed trajectory data across all specified experiments.
    """
    data_all = pd.DataFrame()
    last_part_num = 0
    # Loop for each directory path of position
    for path in pathway_experiment:
        data_exp = pd.DataFrame()
        # Getting the name of the manipulation from the path
        manip_name_search = re.search('ASMOT[0-9]{3}', path)
        if manip_name_search is None:  # If the regex search finds no match, skip to next path
            continue
        manip_name = manip_name_search.group()

        list_files_to_read = [os.path.join(path, f) for f in os.listdir(path)
                              if f.endswith(".hdf5") and name_file in f]
        
        # If no hdf5 files are found, continue to the next directory
        if not list_files_to_read:
            print(f"No HDF5 files found in {path}. Skipping to next directory.")
            continue
        
        print(list_files_to_read)
        list_fields_positions = [re.sub('.hdf5', '', f.replace(path + os.sep, ''))
                                 for f in list_files_to_read]
        # Loop for each file
        for f, position in zip(list_files_to_read, list_fields_positions):
            # Reading data from the file
            try:
                data = pd.read_hdf(f, key='table')
            except ValueError:
                continue

            if name_file == 'features':
                if 'particle' not in data.columns:
                    data = tp.link_df(data,
                                    search_range=search_range,
                                    memory=memory,
                                    neighbor_strategy='KDTree',
                                    link_strategy='auto',
                                    adaptive_stop=adaptive_stop,
                                    )
                
                if 'size' in data.columns and 'mass' in data.columns and 'raw_mass' in data.columns:
                    # data = data[(data['mass'] < min_mass) & (data['size']< max_size) & 
                    # data = data[(data['raw_mass'] > 1000)]
                    mean_mass_by_particle = data.groupby('particle')['raw_mass'].mean()
                    # Filtrer les particules dont la masse moyenne est inférieure à 50000
                    particles_to_keep = mean_mass_by_particle[mean_mass_by_particle > 50000].index
                    # Filtrer 'data' pour ne garder que les lignes correspondant aux particules filtrées
                    data = data[data['particle'].isin(particles_to_keep)]

                                  
            # counts = data.groupby(['particle']).size()
            # particles_to_keep = counts[counts >= nbr_frame_min].reset_index()
            # data = data.merge(particles_to_keep, on=['particle'])

            counts = data.groupby(['particle']).size()
            particles_to_keep = counts[counts >= nbr_frame_min].reset_index(name='count')
            data = data.merge(particles_to_keep, on=['particle'])
            data = data.rename(columns={'particle': 'old_particle'})
            data['particle'], _ = pd.factorize(data['old_particle'])
            data['particle'] += last_part_num
            data["experiment"] = manip_name
            data["position"] = position
            data_exp = pd.concat([data_exp, data])
            
            if len(data_all) != 0:
                last_part_num = data_all['particle'].max() + data_exp['particle'].nunique() + 1
            else:
                last_part_num = data_exp['particle'].nunique()
        if drift:
            # Assuming remove_drift is a previously defined function
            data_exp = remove_drift(traj=data_exp, smooth=2,
                                    save=True, pathway_saving=path,
                                    name=manip_name + '_' + position)
            data_exp = data_exp.drop('frame', axis=1)
            data_exp = data_exp.reset_index(drop=True)
        data_all = pd.concat([data_all, data_exp], ignore_index=True)
        print(manip_name, " : ", data_exp['particle'].nunique())
    data_all['condition'] = condition
    print("Nombre de particules récoltées avant tri: ", data_all['particle'].nunique())
    return data_all


def rolling_mean(datas: pd.DataFrame(), roll: int):
    """
    Apply a rolling mean on datas to smooth mvt.

    Parameters
    ----------
    datas : pd.DataFrame()
        particle trajectories.
    roll : int
        Window on which the roll is applied.

    Returns
    -------
    datas : TYPE
        Datas modified.

    """
    # Application d'une moyenne glissante sur les données de positions
    for i in datas['particle'].unique():
        datas.loc[datas['particle'] == i, 'x'] =\
            datas[datas['particle'] == i]['x'].rolling(window=roll, center=True).median()
        datas.loc[datas['particle'] == i, 'y'] =\
            datas[datas['particle'] == i]['y'].rolling(window=roll, center=True).median()
    datas = datas.dropna()
    return datas

def pixelisation(datas: pd.DataFrame(), size_pix: float):
    """
    Pixelize the trajectories.

    If the particles doesn't move more than one pixel, then we suppose that
    it doesn't move at all.

    Parameters
    ----------
    datas : pd.DataFrame()
        Datas of particles with trajectories.
    size_pix : float
        Pixel size.

    Returns
    -------
    datas : pd.DataFrame()
        DESCRIPTION.

    """
    datas = datas.reset_index()
    datas = datas.drop(columns=['index'])
    for _, group in datas.groupby('particle'):
        x = group['x'].values
        y = group['y'].values
        for i in range(1, len(x)-1):
            # si au temps i la distance entre xi et xi-1 est inférieur à size_ix
            # alors xi = xi-1
            if np.abs(x[i] - x[i-1]) < size_pix:
                x[i] = x[i-1]
            # on ajoute la règle que s'il y'a un écart de 2 pixel entre 2 temps
            # xi-1 et xi et qu'il y'a moins d'un écart de pixel entre xi+1
            # et xi-1, alors en réalité il 'y pas eu déplacement, on pense à un
            # artefact de détéction du centre de masse donc xi = xi-1
            # if np.abs(x[i] - x[i-1]) > 2*size_pix:
            # try:
            #     if np.abs(x[i]-x[i+1]) > 2*size_pix and np.abs(x[i-1]-x[i+1]) < size_pix:
            #         x[i] = x[i-1]
            # except ValueError:
            #     continue
        for i in range(1, len(y)-1):
            if np.abs(y[i] - y[i-1]) < size_pix:
                y[i] = y[i-1]
            # try:
            #     if np.abs(y[i]-y[i+1]) > 2*size_pix and np.abs(y[i-1]-y[i+1]) < size_pix:
            #         y[i] = y[i-1]
            # except ValueError:
            #     continue
        try:
            datas.loc[group.index, 'x'] = x
            datas.loc[group.index, 'y'] = y
        except ValueError:
            continue
    # On remet l'index en place
    datas = datas.dropna()
    datas = datas.reset_index()
    datas = datas.drop(columns=['index'])
    return datas


def keep_nth_image(traj: pd.DataFrame(), n: int, time_frame: int):
    """
    Keep only 1 frame every n frame.

    Cette fonction permet de garder uniquement les valeurs dune image sur nbre_frame.
    L'objectif est de pouvoir faire le tracking sur toutes les images et donc avoir
    une fiabilité du tracking, tout en traitant moins de données qui semblent fausser
    les résultats.

    Parameters
    ----------
    traj : DataFrame
        DataFrame of the trajectories.
    n: int
        1/nbre frame will be kept.

    Returns
    -------
    df : DataFrame
        DESCRIPTION.

    """
    d_f = traj.query('frame%'+str(n)+'==0')
    d_f['frame'], _ = pd.factorize(d_f['frame'])
    d_f = d_f.reset_index()
    d_f = d_f.drop(columns='index', axis=1)
    time_frame = n*time_frame
    return d_f, time_frame


def vit_instant_new(traj, lag_time, pix_size, triage):
    """
    Compute the instant speed of each particle.

    Parameters
    ----------
    traj : pandas DataFrame
        DataFrame containing the trajectory data.
    lag_time : float
        Lag time between consecutive time steps.
    pix_size : float
        Pixel size conversion factor.
    triage : float
        Triage value.

    Returns
    -------
    traj : pandas DataFrame
        Updated DataFrame with the 'VitInst' column.

    """
    traj_copy = traj.copy()  # Make a copy of the DataFrame to avoid modifying the original data

    # Drop the 'VitInst' column if it exists, ignoring errors
    traj_copy = traj_copy.drop(['VitInst', 'dx [pix]', 'dy [pix]',
                                'displacement [pix]'], axis=1, errors='ignore')

    # Calculate the differences between coordinates
    traj_copy['dx [pix]'] = traj_copy.groupby('particle')['x'].diff()
    traj_copy['dy [pix]'] = traj_copy.groupby('particle')['y'].diff()

    # Compute the displacement
    traj_copy['displacement [pix]'] = traj_copy[
        'dx [pix]'].apply(lambda x: x**2) + traj_copy['dy [pix]'].apply(lambda y: y**2)
    traj_copy['displacement [pix]'] = traj_copy[
        'displacement [pix]'].apply(lambda x: np.sqrt(x))
    # Calculate the duration between time steps
    delta_t = triage * (lag_time / 60) * traj_copy.groupby('particle')['frame'].diff()

    # Calculate the instant speed in µm/min
    traj_copy['VitInst [um/min]'] = traj_copy['displacement [pix]'] * pix_size / delta_t
    return traj_copy

def calculate_total_path_first_frames(dataframe, first_n_frames=10):
    """
    Calculate the total path length traveled by each particle over its first N frames.

    Parameters:
    - dataframe : pd.DataFrame
        DataFrame containing tracking data with at least 'x', 'y', 'frame', and 'particle' columns.
    - first_n_frames : int, optional
        Number of first frames to consider for each particle (default is 10).

    Returns:
    - dataframe : pd.DataFrame
        DataFrame with an added column for total path length over the first N frames.
    """
    # Sorting the dataframe
    dataframe_sorted = dataframe.sort_values(by=['particle', 'frame'])
    
    # Function to calculate path length for the first N frames of each particle
    def total_path_first_n_frames(group):
        if len(group) < 2:
            return 0
        group_first_n = group.head(first_n_frames)
        dx = np.diff(group_first_n['x'])
        dy = np.diff(group_first_n['y'])
        return np.sum(np.sqrt(dx**2 + dy**2))

    # Applying the function to each group
    total_paths = dataframe_sorted.groupby('particle').apply(total_path_first_n_frames)

    # Mapping total path lengths back to the original dataframe
    dataframe['total_path_first_n'] = dataframe['particle'].map(total_paths)

    return dataframe

def center(traj):
    """
    Centers the trajectory data in the (x, y) plane.

    Parameters
    ----------
    Traj : Pandas DataFrame
        DataFrame containing the trajectory data.
        The DataFrame must have columns 'x', 'y', and 'particle'.

    Returns
    -------
    Traj : Pandas DataFrame
        DataFrame with the original 'x' and 'y' columns and two new columns:
        'X0': the initial x-position for each particle.
        'Y0': the initial y-position for each particle.
        'Xc': the centered x-position for each particle (x - X0).
        'Yc': the centered y-position for each particle (y - Y0).
    """
    # Check if the required columns are present in the DataFrame
    if 'x' not in traj.columns or 'y' not in traj.columns or 'particle' not in traj.columns:
        raise ValueError("The DataFrame must have columns 'x', 'y', and 'particle'.")
    # Check if the 'x' and 'y' columns contain numeric values
    if not pd.api.types.is_numeric_dtype(traj['x']) or not pd.api.types.is_numeric_dtype(traj['y']):
        raise ValueError("The 'x' and 'y' columns must contain numeric values.")
    # Create a dictionary of initial x-positions for each particle
    x_zero = {k: df['x'].values[0] for k, df in traj.groupby('particle')}
    # Create a dictionary of initial y-positions for each particle
    y_zero = {k: df['y'].values[0] for k, df in traj.groupby('particle')}
    # Add a column 'X0' to Traj with the initial x-position for each particle
    traj['X0 [pix]'] = [x_zero[part] for part in traj['particle']]
    # Add a column 'Y0' to Traj with the initial y-position for each particle
    traj['Y0 [pix]'] = [y_zero[part] for part in traj['particle']]
    # Add a column 'Xc' to Traj with the centered x-position for each particle
    traj['Xc [pix]'] = traj['x'] - traj['X0 [pix]']
    # Add a column 'Yc' to Traj with the centered y-position for each particle
    traj['Yc [pix]'] = traj['y'] - traj['Y0 [pix]']
    # Supp X0 et Y0
    traj = traj.drop(['X0 [pix]', 'Y0 [pix]'], axis=1)
    return traj

def length_displacement(traj, size_pix, lag_time=15, triage=1):
    """
    Déplacement de la particule.

    Renvoie le déplacement absolue (start-end) et le déplacement intégrée des particules.

    Parameters
    ----------
    traj : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if 'dx [pix]' not in traj.columns:
        traj = vit_instant_new(traj=traj, lag_time=lag_time, pix_size=size_pix, triage=triage)
    traj['cumulative displacement [um]'] = traj.groupby('particle')['displacement [pix]'].cumsum()*size_pix
    # Grouper par "particle" et appliquer la fonction personnalisée
    start_end = pd.DataFrame()
    start_end['start-end [um]'] = size_pix * np.sqrt(
        traj.groupby('particle').apply(subtract_first_last, 'x')**2 +
        traj.groupby('particle').apply(subtract_first_last, 'y')**2)
    return traj, start_end

def traj_clustering_with_fit_cutoff(df: pd.DataFrame,
                                    imsd: pd.DataFrame,
                                    hist: bool,
                                    lag_time_fit: int,
                                    micronperpixel: float,
                                    fps: float,
                                    binsize: int = 300,
                                    peak_height: int = 5,
                                    peak_width: int = 1,
                                    save: bool = True,
                                    pathway_fig: Optional[str] = None,
                                    name: Optional[str] = None,
                                    img_type: str = "jpg",
                                    plot: bool = True,
                                    color_sup_inf: Tuple[str, str] = ('red', 'blue'),
                                    cutoff_default: Optional[float] = None
                                    ) -> Tuple[List[float], List[float], List[int], List[int], float]:
    """
    Clusters trajectories based on the slope of their mean square displacement (MSD).

    This function separates particle trajectories into 'fast' and 'slow' groups
    by analyzing the slopes of their MSDs.
    It involves fitting the slopes to a bimodal distribution and finding a
    cutoff point that differentiates the two clusters.

    Parameters
    ----------
    -df (pd.DataFrame): DataFrame
        containing the original tracked particle data.
    -imsd (pd.DataFrame): DataFrame
        of calculated individual MSDs for each particle.
    -hist (bool): Whether or not to display a histogram of the MSD slopes.
    -lag_time_fit (int): The time over which the MSD was calculated.
    -micronperpixel (float): Conversion factor for pixels to microns.
    -fps (float): Frames per second of the original tracking video/data.
    -binsize (int, optional): The number of bins for the histogram. Defaults to 300.
    -peak_height (int, optional): Minimum height of peaks to be identified in the histogram.
            Defaults to 5.
    -peak_width (int, optional): Minimum width of peaks to be identified in the histogram.
            Defaults to 1.
    -save (bool, optional): Whether to save generated plots.
            Defaults to True.
    -pathway_fig (Optional[str], optional): Path to save the figures if `save` is True.
            Defaults to None.
    -name (Optional[str], optional): Base name for saving plots.
            Defaults to None.
    -img_type (str, optional): File type for saving images (e.g., 'jpg', 'png').
            Defaults to "jpg".
    -plot (bool, optional): Whether to generate plots.
            Defaults to True.
    -color_sup_inf (Tuple[str, str], optional): Colors for differentiating 'fast'
    and 'slow' in plots.
            Defaults to ('red', 'blue').
    -cutoff_default (Optional[float], optional): Predefined cutoff value for
    slope separation. If None, the cutoff is calculated.
            Defaults to None.

    Returns
    -------
    Tuple[List[float], List[float], List[int], List[int], float]: Returns five items:
        - List of slopes for 'slow' particles
        - List of slopes for 'fast' particles
        - List of particle IDs for 'slow' particles
        - List of particle IDs for 'fast' particles
        - Calculated or predefined cutoff value for slope separation
    """
    S = []
    Parts = []
    negative_parts = []
    S_slow, S_fast = [], []
    Parts_slow, Parts_fast = [], []
    EPSILON = 1e-15  # Valeur epsilon pour éviter la division par zéro
    bin_size = binsize

    # Pré-calcule Logarithme
    log_index = np.log10(imsd.index[0:lag_time_fit] + EPSILON)
    log_imsd = np.log10(imsd.iloc[0:lag_time_fit] + EPSILON)

    positive_mask = np.zeros(len(imsd.columns), dtype=bool)
    S = []
    for idx, col in enumerate(imsd.columns):
        s, _, _, _, _ = stats.linregress(log_index, log_imsd[col])
        if s >= 0:
            S.append(s)
            positive_mask[idx] = True  # Utilisez idx au lieu de get_loc pour plus d'efficacité

    # Parts et negative_parts
    Parts = imsd.columns[positive_mask].tolist()
    negative_parts = imsd.columns[~positive_mask].tolist()

    counts, bins = np.histogram(S, bins=bin_size)
    counts[2:len(counts)-2] = (counts[0:len(counts)-4] + counts[1:len(counts)-3] +
                               counts[2:len(counts)-2] + counts[3:len(counts)-1] +
                               counts[4:len(counts)])/5

    peaks, _ = find_peaks(counts, height=peak_height, width=peak_width)  # prominence=2,distance=50)

    if len(peaks) > 1:
        # déterminer les bornes des pics--> j'ai enlevé le width au dessus qui était =4
        left, right = np.searchsorted(bins, [bins[peaks[0]], bins[peaks[1]]])
        # trouver le minimum entre les pics
        min_between_peaks = bins[left + np.argmin(counts[left:right])]
        # Ajustement de la fonction bimodale aux données
        popt, pcov = curve_fit(bimodal, bins[0:bin_size], counts,
                               p0=[counts[peaks[0]], bins[peaks[0]],
                                   abs(min_between_peaks - bins[peaks[0]]),
                                   counts[peaks[1]], bins[peaks[1]],
                                   abs(min_between_peaks - bins[peaks[1]])], maxfev=1000000)

        def f(x):
            """
            Return bimodal function of x.

            Parameters
            ----------
            x : float

            Returns
            -------
            bimodal(x): float
                DESCRIPTION.

            """
            return bimodal(x, *popt)
        # Find the minimum using quadratic inverse interpolation
        res = minimize_scalar(f, bounds=[0.2, 1.0], method='Bounded')
        # Print the result
        print("Minimum local: ", res.x)
        cutoff = res.x
    else:
        cutoff = 0
    if cutoff_default is not None:
        cutoff = cutoff_default
    Parts_fast = [i for i, s in zip(Parts, S) if cutoff == 0 or s >= cutoff]
    S_fast = [s for s in S if cutoff == 0 or s >= cutoff]

    Parts_slow = [i for i, s in zip(Parts, S) if s < cutoff]
    S_slow = [s for s in S if s < cutoff]

    print('# negative slope', len(negative_parts))

    if plot:
        #  First plot is to present the fit
        fig, axis = plt.subplots(figsize=(20, 20))
        # Largeur des bins
        width = bins[1] - bins[0]
        # Tracer le bar plot
        axis.bar(bins[:-1], counts, align='edge', width=width, color='green', alpha=0.5)
        axis.plot(bins[0:bin_size], counts, label="Smoothed histo")
        # Afficher le plot
        if len(peaks) > 1:
            # Plot
            axis.plot([bins[peaks[0]], bins[peaks[0]]], [0, np.max(counts)],
                      label=f'Peak 1 = {round(bins[peaks[0]], 2)}')
            axis.plot([bins[peaks[1]], bins[peaks[1]]], [0, np.max(counts)],
                      label=f'Peak 2 = {round(bins[peaks[1]], 2)}')
            axis.plot([min_between_peaks, min_between_peaks], [0, np.max(counts)],
                      color='red', label=f'Min {round(min_between_peaks,2)}')
            axis.plot([cutoff, cutoff], [0, np.max(counts)],
                      label=f'min bimodal = {round(cutoff,2)}', color='blue')
            axis.plot(bins[0:bin_size], bimodal(bins[0:bin_size], *popt), 'r-', label='Fit')
            axis.legend(fontsize=30)
            axis.tick_params(axis='both', which='major', labelsize=20)

        plt.title("Slopes of MSD", fontsize=40)
        plt.xlabel(xlabel="slopes", fontsize=30)
        plt.ylabel(ylabel="Count", fontsize=30)
        plt.grid()
        plt.show()
        # Adjust the spacing of the plot
        fig.tight_layout()
        if save:
            fig.savefig(os.path.join(pathway_fig, f"compute_cutoff.{img_type}"), format=img_type)

        #  NEW PLOT FOR sorted cells
        Data_slow = df[df['particle'].isin(Parts_slow)]
        Data_fast = df[df['particle'].isin(Parts_fast)]

        # Ax[0].plot(range(0,10), range(0,10), color = 'red', alpha = 0.2)
        if not Data_slow.empty:
            IM_slow = imsd[Parts_slow]
        else:
            IM_slow = pd.DataFrame()
        if not Data_fast.empty:
            IM_fast = imsd[Parts_fast]
        else:
            IM_fast = pd.DataFrame()
        fig, Ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
        # Plot the datas
        if not Data_slow.empty:
            Ax[0].plot(IM_slow.index.values, IM_slow.values, color=color_sup_inf[1],
                       alpha=0.1, linewidth=0.1)
        if not Data_fast.empty:
            Ax[0].plot(IM_fast.index.values, IM_fast.values, color=color_sup_inf[0],
                       alpha=0.1, linewidth=0.1)
        Ax[0].set_xscale("log")
        Ax[0].set_yscale("log")
        Ax[0].set_xlim(15, 70)
        Ax[0].set_ylim(0.01, 1000)
        Ax[0].set_title("n="+str(len(S)), y=1.0, pad=-8, fontsize=15)
        Ax[0].set_xlabel('lag time [s]', fontsize=20)
        Ax[0].set_ylabel('IMSD [$\\mu$m$^2$]', fontsize=20)
        Ax[0].tick_params(axis='both', which='major', labelsize=15)

        if hist:
            ax = Ax[1]
            ax.hist(S, bins=250, density=True, label=[])
            ax.tick_params(axis='both', which='major', labelsize=15)
            for bar in ax.containers[0]:
                x = bar.get_x() + 0.5 * bar.get_width()
                if x < cutoff:
                    bar.set_color(color_sup_inf[1])
                elif x > cutoff:
                    bar.set_color(color_sup_inf[0])
            ax.set_xlabel('slope value', fontsize=20)
            ax.set_ylabel('count', fontsize=20)
        # Ax[0,1].set_xlim(0,2)
        # Ax[0,1].set_ylim(0,2)
        # Ax[0,1].set_xlabel('IMSD slope')
        # Ax[0,1].set_ylabel('Normalized Counts')
        fig.suptitle(f"Results for {name}", fontsize=25)
        fig.tight_layout()
        if save:
            fig.savefig(os.path.join(pathway_fig, f'MSD_slopes_{name}.{img_type}'), format=img_type)
    return S_slow, S_fast, Parts_slow, Parts_fast, cutoff

def plot_datas(x_values: List[Union[float, int]],
               y_values: List[Union[float, int]],
               title: str,
               x_label: Optional[str] = None,
               y_label: Optional[str] = None,
               x_lim: Optional[Union[tuple, list]] = None,
               y_lim: Optional[Union[tuple, list]] = None,
               save: bool = False,
               path_save_pic: Optional[str] = None,
               parameters_plot: Optional[dict] = {'color': 'green',
                                                  'linewidth': 0.5,
                                                  'linestyle': 'solid'},
               img_type: str = "jpg") -> None:
    """
    Plot data with the option to save the resulting figure, creating directories if needed.

    Parameters are as described in the original code, with added functionality
    for handling missing directories.
    """
    # Create a new figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a line plot of y_values against x_values
    ax.plot(x_values, y_values, color=parameters_plot['color'],
            linewidth=parameters_plot['linewidth'],
            linestyle=parameters_plot['linestyle'])

    # Set the label for the x-axis if provided
    if x_label is not None:
        ax.set_xlabel(x_label)

    # Set the label for the y-axis if provided
    if y_label is not None:
        ax.set_ylabel(y_label)

    # Set the title of the plot
    ax.set_title(title)

    # Set the limits of the x-axis if provided
    if x_lim is not None:
        ax.set_xlim(x_lim)

    # Set the limits of the y-axis if provided
    if y_lim is not None:
        ax.set_ylim(y_lim)

    # Display the plot
    plt.show()

    # Save the figure if the save flag is True
    if save:
        if path_save_pic is None:
            raise ValueError("The saving path is required.")
        else:
            # Ensure the directory exists
            os.makedirs(path_save_pic, exist_ok=True)
             # Clean the title to create a valid filename
            filename = f"{title}.{img_type}"
            # Replace invalid characters with underscores
            filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
            # Additionally, replace spaces with underscores if desired
            filename = filename.replace(" ", "_")
            # Remove any remaining problematic characters
            filename = filename.replace('/', '_')
            # Save the figure
             # Build the full path
            save_path = os.path.join(path_save_pic, filename)
            # Save the figure
            fig.savefig(save_path, format=img_type)
            print(f"Figure saved at: {save_path}")
 
def plot_msd(msd, fps, name="MSD of all frames in function of lag time (s)",
             color_plot: str = 'red', save=False, pathway_saving=None,
             alpha=0.05, linewidth=0.01, img_type='jpg'):
    """
    Plots the mean square displacement (MSD) of particle trajectories.

    Parameters:
    - msd : pd.DataFrame
        DataFrame containing MSD values with index as lag times.
    - fps : float
        Frames per second, used to calculate time scale.
    - name : str, optional
        Title for the plot. Defaults to "MSD of all frames in function of lag time (s)".
    - color_plot : str, optional
        Color of the plot line.
    - save : bool, optional
        If True, the plot will be saved to the specified path.
    - pathway_saving : str, optional
        Path where the plot will be saved if `save` is True.
    - alpha : float, optional
        Transparency of the plot line.
    - linewidth : float, optional
        Width of the plot line.
    - img_type : str, optional
        Image file format for saving the plot.

    Returns:
    - None
    """
    # Get the number of curves from the number of columns in the MSD DataFrame
    nbr_curves = len(msd.columns)

    # # Set the index
    # msd = msd.set_index("lag time [s]")

    # Create a new figure and axis object
    fig, axis = plt.subplots(figsize=(20, 20))

    # Plot the MSD data on the axis object
    axis.plot(msd, alpha=alpha, linewidth=linewidth, color=color_plot)

    # Set the limits of the x-axis and y-axis
    axis.set_xlim([1 / fps, 100 / fps])
    axis.set_ylim(0.01, 10000)

    # Set the x-axis and y-axis to be on a log scale
    axis.set(xscale="log", yscale="log")

    # Set the x-axis label
    axis.set_xlabel("lag time (s)", fontsize=30)

    # Set the x-axis label
    axis.set_ylabel("MSD", fontsize=30)

    # Add a text box to the plot with the number of curves
    textstr = f"nbre curves: {nbr_curves}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    axis.text(0.05, 0.95, textstr, transform=axis.transAxes, fontsize=30,
              verticalalignment="top", bbox=props)

    axis.tick_params(axis='both', which='major', labelsize=20)

    # Set the title of the plot
    fig.suptitle(name, fontsize=40, fontweight="bold", fontstyle='italic', fontname="Arial")

    # Adjust the spacing of the plot
    fig.tight_layout()

    # Save the plot if the "save" parameter is True
    if save:
        fig.savefig(os.path.join(pathway_saving, f"{name}.{img_type}"), format=img_type)

def plot_displacement(traj: pd.DataFrame, start_end: pd.DataFrame,
                      color_plot: str = 'green', linewidth: float = 0.1,
                      alpha: float = 0.1, save: bool = False, xlim: list = None, ylim: list = None,
                      pathway_saving: str = None, name: str = None, img_type: str = 'jpg'):
    """
    Plot displacement vs time and start-end vs cumulative displacement.

    Parameters:
    traj : pd.DataFrame - DataFrame containing trajectory data with 'time (min)' and 'cumulative displacement [um]'.
    start_end : pd.DataFrame - DataFrame to store and display start-end information.
    color_plot : str - Color for the plot lines and points.
    linewidth : float - Line width for plot lines.
    alpha : float - Alpha transparency for lines and markers.
    save : bool - Flag to save the generated plot.
    xlim : list - X-axis limits for the plot.
    ylim : list - Y-axis limits for the plot.
    pathway_saving : str - Path to save the plot files.
    name : str - Base name for saved plot files.
    img_type : str - Image file type for saved plots.
    """
    if traj.empty:
        print("Input trajectory dataframe is empty, unable to plot")
        return

    # Plot cumulative displacement vs time
    fig, ax = plt.subplots(figsize=(10, 10))
    grouped = traj.groupby('particle')
    for name, group in grouped:
        adjusted_time = group['time (min)'] - group['time (min)'].iloc[0]
        ax.plot(adjusted_time, group['cumulative displacement [um]'], 
                alpha=alpha, linewidth=linewidth, color=color_plot, label=name)
    ax.set_xlabel('Time (min)', fontsize=20)
    ax.set_ylabel('Cumulative displacement [um]', fontsize=20)
    plt.title("Cumulative Displacement vs Time", fontsize=20, fontweight="bold", fontstyle='italic', fontname="Arial")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    if save and pathway_saving and name:
        fig_path = os.path.join(pathway_saving, f"cumulative_displacement_vs_time_{name}.{img_type}")
        fig.savefig(fig_path, format=img_type)

    # Add the cumulative displacement to start_end DataFrame
    cumulative = traj.groupby('particle')['cumulative displacement [um]'].last()
    start_end_combined = pd.concat([start_end, cumulative], axis=1)

    # Plot start-end vs cumulative displacement
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(start_end_combined['cumulative displacement [um]'], start_end_combined['start-end [um]'],
               marker='+', linewidth=0.5, alpha=alpha, color=color_plot)
    ax.set_xlabel('Cumulative Displacement (um)', fontsize=20)
    ax.set_ylabel('Start-End Displacement (um)', fontsize=20)
    plt.title("Start-End Displacement vs Cumulative Displacement", fontsize=20, 
              fontweight="bold", fontstyle='italic', fontname="Arial")
    plt.tight_layout()
    plt.show()
    if save and pathway_saving and name:
        fig_path = os.path.join(pathway_saving, f"start_end_vs_cumulative_{name}.{img_type}")
        fig.savefig(fig_path, format=img_type)

def plot_displacement_low_and_high(traj_sup: pd.DataFrame, traj_inf: pd.DataFrame, part_coef_sup, part_coef_inf,
                                   start_end: pd.DataFrame, alpha: float=0.1, linewidth: float=0.1,
                                   xlim: list=None, ylim: list=None, 
                                   color_sup_inf: tuple=('red', 'blue'), save=False,
                                   pathway_saving=None, name=None, img_type="jpg"):
    if traj_sup.empty and traj_inf.empty:
        print("Both input dataframes are empty, unable to plot")
        return

    # Plotting cumulative displacement vs time
    fig, ax = plt.subplots(figsize=(10, 10))
    def plot_group(grouped_df, color, label_prefix):
        for name, group in grouped_df:
            adjusted_time = group['time (min)'] - group['time (min)'].iloc[0]
            ax.plot(adjusted_time, group['cumulative displacement [um]'], 
                    alpha=alpha, linewidth=linewidth, color=color, 
                    label=f"{label_prefix} {name}")

    if not traj_sup.empty:
        grouped_sup = traj_sup.groupby('particle')
        plot_group(grouped_sup, color_sup_inf[0], 'High')
    
    if not traj_inf.empty:
        grouped_inf = traj_inf.groupby('particle')
        plot_group(grouped_inf, color_sup_inf[1], 'Low')

    ax.set_xlabel('Time (min)', fontsize=20)
    ax.set_ylabel('Cumulative displacement [um]', fontsize=20)
    ax.set_title(f"Cumulative Displacement vs Time ({name})", fontsize=20, 
                 fontweight="bold", fontstyle='italic', fontname="Arial")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    if save:
        if not pathway_saving:
            pathway_saving = './'
        fig.savefig(os.path.join(pathway_saving, f"cumulative_displacement_vs_time_{name}.{img_type}"), format=img_type)

    # Handling start-end vs cumulative displacement
    start_end_sup = start_end.loc[start_end.index.isin(part_coef_sup)]
    cumulative_sup = traj_sup.groupby('particle')['cumulative displacement [um]'].last()
    start_end_sup = start_end_sup.assign(cumulative=cumulative_sup)
    start_end_inf = start_end.loc[start_end.index.isin(part_coef_inf)]
    cumulative_inf = traj_inf.groupby('particle')['cumulative displacement [um]'].last()
    start_end_inf = start_end_inf.assign(cumulative=cumulative_inf)

    # Plotting start-end vs cumulative displacement
    if not start_end_sup.empty and not start_end_inf.empty:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(start_end_sup['cumulative'], start_end_sup['start-end [um]'],
                   marker='+', linewidth=0.8, alpha=alpha, color='blue')
        ax.scatter(start_end_inf['cumulative'], start_end_inf['start-end [um]'],
                   marker='+', linewidth=0.8, alpha=alpha, color='red')
        ax.set_xlabel('Cumulative displacement (um)', fontsize=20)
        ax.set_ylabel('Start-End displacement (um)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.title(f'Start-End Displacement vs Cumulative Displacement', fontsize=20,
                  fontweight="bold", fontstyle='italic', fontname="Arial")
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()
        if save:
            fig.savefig(os.path.join(pathway_saving, f"start_end_vs_cumulative_{name}.{img_type}"),
                        format=img_type)
