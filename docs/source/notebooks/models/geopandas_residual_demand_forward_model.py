import geopandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
# download the respective shp file from e.g. 
# https://www.suche-postleitzahl.org/
# and specify the file location below
# A large part of the below code is taken from the post https://juanitorduz.github.io/germany_plots/
fp = "./geo_data/plz-3stellig.shp"
def plot_efficiency_map(results, timestep: int = -1, path: int = 0):
    """
    Plot the wind efficiency map of Germany.
    """
    map_df = geopandas.read_file(fp) # read the geometry
    np.random.seed(42)
    x = np.empty((3,2,))
    x[:,0]=np.array([9.993682, 11.576124, 13.404954]) #Hamburg,Munich,Berlin
    x[:,1]=np.array([53.551086, 48.137154, 52.520008]) #Hamburg,Munich,Berlin
    fx = np.array([results.get('Region_'+str(i)+'_FWD0')[timestep,path] for i in range(3)]) # we use 
    interp = RBFInterpolator(x, d=fx) # Simple interpolation with RBF for illustration purposes. This may lead to inconsistent results.
    x_interp = np.empty((map_df.shape[0],2))
    centroids = map_df['geometry'].to_crs('epsg:3785').centroid.to_crs(map_df.crs)
    x_interp[:,0] = centroids.x#map_df['geometry'].centroid.x
    x_interp[:,1] = centroids.y#map_df['geometry'].centroid.y
    map_df['wind'] = interp(x_interp)

    ax = plt.subplot()
    map_df.plot('wind', legend=True, ax=ax, vmin=0.0, vmax=1.0, legend_kwds={'label': "Wind Efficiency"})#,'orientation': "horizontal"})
    top_cities = {
        'Berlin': (13.404954, 52.520008), 
        'Köln': (6.953101, 50.935173),
        'Düsseldorf': (6.782048, 51.227144),
        'Frankfurt am Main': (8.682127, 50.110924),
        'Hamburg': (9.993682, 53.551086),
        'Leipzig': (12.387772, 51.343479),
        'München': (11.576124, 48.137154),
        'Dortmund': (7.468554, 51.513400),
        'Stuttgart': (9.181332, 48.777128),
        'Nürnberg': (11.077438, 49.449820),
        'Hannover': (9.73322, 52.37052)
    }
    for c in top_cities.keys():
        # Plot city name.
        ax.text(
            x=top_cities[c][0], 
            # Add small shift to avoid overlap with point.
            y=top_cities[c][1] + 0.08, 
            s=c, 
            fontsize=12,
            ha='center', 
        )
        # Plot city location centroid.
        ax.plot(
            top_cities[c][0], 
            top_cities[c][1], 
            marker='o',
            c='black', 
            alpha=0.5
        )

    ax.set(
        title='One Realization of Simulated Wind Efficiency in Germany', 
        aspect=1.3, 
        facecolor='lightblue'
    );
    #frame1 = plt.gca()
    #frame1.axes.get_xaxis().set_visible(False)
    #frame1.axes.get_yaxis().set_visible(False)

    #plt.savefig('wind_efficiency_map.png',dpi=1500)