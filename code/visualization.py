# --------------------------------------------------------------
# This file contains several useful visualization functions that
# can be applied on pandas dataframes
#
# 2021 Frédéric Berdoz, Zurich, Switzerland
# --------------------------------------------------------------

# Data processing
import pandas as pd
import numpy as np

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def empty_cells(df, savepath=None, dpi=100):
    """Plot a map of the empty cells in a pandas dataframe.
    
    Arguments:
        - df: Dataframe containing the data.
        - savepath: path (including filename) where to store the plot (not stored if 'None' is passed).
        - dpi: Resolutin of the figure (dot per inch).
    """
    
    # Create figures
    plt.figure(figsize=(df.shape[1]/5, df.shape[0]/5), dpi=dpi)
    
    # Plot heatmap
    nan_map = sns.heatmap(df.isna(),
                          cbar=False,
                          cmap="YlGnBu",
                          linewidths = 0.1,
                          linecolor = "grey")

    # Labels on top
    plt.tick_params(axis='x', which='major', labelbottom=False, 
                    bottom=False, top=True, labeltop=True, labelrotation=90)
    plt.tick_params(axis='y', which='major', left = True)
    
    # Save the figure at given location
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
        
def correlation_matrix(corr_matrix, savepath=None, title="", dpi=100, 
                       sep_lines=None, triu_mask=False, clip=True, 
                       cmap="RdBu", annot=True, cbar=True):
    """Plot a heatmap of a correlation matrix.
    
    Arguments:
        - corr_matrix: Dataframe containing the correlation matrix of the data.
        - savepath: Path (including filename) where to store the plot (not stored if 'None' is passed).
        - title: Title for the plot.
        - dpi: Resolutin of the figure (dot per inch).
        - sep_lines: Index of columns and rows after which a seperatlion line shoulf be displayed.
        - triu_mas: Boolean. Decide if the upper triangle of the matrix should be displayed.
        - clip: Boolean. Decide if the color map should be clipped to the interval [-1, 1].
        - cmap: Color map name (string).
        - annot: Boolean. Decide if the exact value of the correlation should be displayed in each cell.
        - cbar: Boolean. Decide if the color bar reference should be displayed.
    """
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(corr_matrix.shape[1]/3, 
                                          corr_matrix.shape[0]/3), dpi=dpi)
    
    # Create mask (label on top if no mask)
    plt.tick_params(axis='y', which='major', left = True)
    mask = np.zeros_like(corr_matrix)
    if triu_mask:
        mask = np.zeros_like(corr_matrix)
        mask[np.triu_indices_from(mask, k=1)] = True
        
    if clip:
        vmin =-0.999
        vmax = 0.999
    else:
        vmin=None
        vmax=None
    
    # Title
    ax.set_title(title)
    # Plot heatmap
    corr_map = sns.heatmap(corr_matrix,
                           cbar=cbar,
                           cmap=cmap,
                           cbar_kws= {"shrink" : 0.5},
                           linewidth=0.1,
                           annot=annot, 
                           annot_kws = {"fontsize" : 6},
                           fmt=".2f",
                           mask=mask,
                           square=True,
                           vmin=vmin,
                           vmax=vmax,
                           ax=ax)
    
   # Draw separation line to distinguish groups of variables
    if sep_lines is not None:
        corr_map.hlines(sep_lines, *corr_map.get_xlim(), color="k", linewidth=3)
        corr_map.vlines(sep_lines, *corr_map.get_ylim(), color="k", linewidth=3)
      
    # Save the plot at given location
    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')

def correlation_plot(df, savepath=None, title="", hue=None):
        """Visualization of the correlation between the variables.
            
    Arguments:
        - df: Dataframe containing the data.
        - savepath: Path (including filename) where to store the plot (not stored if 'None' is passed).
        - title: Title for the plot.
        - hue: Hue variable (categorical).
    """
        
        # Plot
        g = sns.PairGrid(df, vars=[col for col in df.columns if col != hue],
                         diag_sharey=False, hue=hue)
        g.map_upper(sns.regplot, scatter_kws={'s': 10})
        g.map_diag(sns.distplot)
        g.map_lower(sns.kdeplot, shade=True, shade_lowest=False)
        
        # Title and legend
        g.fig.suptitle(title, y=1.02)
        
        # Add legend if hue variable has more than one category
        if hue is not None:
            if df[hue].nunique() > 1:
                 g.add_legend()
            
        # Save the plot at given location
        if savepath is not None:
            g.savefig(savepath, bbox_inches='tight') 
            
def histogram(df, savepath=None, title="", which="both", hue=None, **hist_kwags):
    """Histogram (and/or cumulative histogram) of the data.
                
    Arguments:
        - df: Dataframe containing the data.
        - savepath: Path (including filename) where to store the plot (not stored if 'None' is passed).
        - title: Title for the plot.
        - which: Either 'hist' (histogram), 'cum' (cumulative) or 'both'.
        - hue: Hue variable (categorical).
        - **hist_kwags: Arguments passed to the 'hist' function.
    """
    
    # Check argument type and value
    if not isinstance(df, pd.core.frame.DataFrame):
        raise TypeError("First positional argument must be a pandas DataFrame. '{}' was given.".format(type(df)))
    if hue is not None and hue not in df:
        raise ValueError("Hue '{}' is not a valid variable. Valid options are {}".format(hue, df.columns))
    if which not in ["hist", "both", "both"]:
        raise ValueError("'which' must be either 'hist', 'cum' or 'both'. '{}'' was given.".format(which))
    
    # Figure creation
    nrows = df.shape[1] - int(hue in df)
    ncols = 2 if which == "both" else 1
    fig, axs  = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 3), squeeze=False)
    
    # Iterate through variables
    i = 0
    for col in df:
        if col != hue:
            # Ax selection
            if which == "hist":
                ax_hist = axs[i, 0]
            elif which == "cum":
                ax_cum = axs[i, 0]
            elif which == "both":
                ax_hist = axs[i, 0]
                ax_cum = axs[i, 1]
            
            
            # Plot
            if which in ["hist", "both"]:
                #ax_hist = sns.histplot(df, x=col, hue=hue, ax=ax_hist, **hist_kwags) # for sns 11.x
                if hue is not None:
                    ax_hist.hist([df[df[hue]==value][col] for value in df[hue].unique()],
                                 label=[value for value in df[hue].unique()], **hist_kwags)
                    ax_hist.legend(loc="upper right")
                else:
                    ax_hist.hist(df[col], **hist_kwags)
                    
                ax_hist.text(x=0.02, y=0.9, s=col, transform = ax_hist.transAxes)
                if i == 0:
                    ax_hist.set_title(title + ": Histogram" + int(nrows>1)*"s")
                
                ax_hist.grid(True)
                
            if which in ["cum", "both"]:
                #ax_cum = sns.distplot(df, x=col, hue=hue, ax=ax_hist, cumulative=True, **hist_kwags)  # for sns 11.x
                if hue is not None:
                    ax_cum.hist([df[df[hue]==value][col] for value in df[hue].unique()],
                                 label=[value for value in df[hue].unique()], cumulative=True, **hist_kwags)
                    ax_cum.legend(loc="upper right")
                else:
                    ax_cum.hist(df[col], cumulative=True, **hist_kwags)
                    
                ax_cum.text(x=0.02, y=0.9, s=col, transform = ax_cum.transAxes)
                if i == 0:
                    ax_cum.set_title(title + ": Cumulative histogram" + int(nrows>1)*"s")
                
                ax_cum.grid(True)   
            i += 1
     # Save the plot at given location
    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')                               
                                        