# --------------------------------------------------------------
# This file contains the code for the database objects used to 
# analyse the data of different mechanisms,
#
# 2021 Frédéric Berdoz, Zurich, Switzerland
# --------------------------------------------------------------

# Data processing
import pandas as pd
import numpy as np
import xlrd

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Data processing
from scipy.fftpack import fft

# Custom package
from helpers.visualization import correlation_matrix, correlation_plot, histogram

# System
import os

class AbstractDb:
    """Parent (blueprint) class for different data analysis. Encapsulate the code for the data visualization."""
    
    def __init__(self):
        """Class constructor."""
        self.model = None
        self.db = None
        self.db_original = None
        self.files = []
        self.db_dir = None
        
        # Torch dataset and dataloader
        self.ds = None
        self.dl = None
        
    def load_data(self):
        """[Abstract] Load the data."""
        raise NotImplementedError
        
    def _get_model_str(self, mode):
        """[Abstract] Return a string for the model name."""
        raise NotImplementedError
        
    def restore(self):
        """Restore to the raw data (not pre-processed)"""
        self.db = self.db_original.copy()
    
    def save_db(self, path="./exports", extension="xls"):
        """Save the database (pandas dataframe) in an excel file.
        
        Arguments:
            - path: Directoy where the data should be exported.
            - extension: Extension (deciding the format) for the export.
        """
        
        # Create directory for the db
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save database
        if self.db is not None:
            if extension[:2] == "xl":
                self.db.to_excel(path + "/DataBaseExport_" + self._get_model_str(mode="filename") + "." + extension, "Database")
            elif extension == "csv":
                self.db.to_csv(path + "/DataBaseExport_" + self._get_model_str(mode="filename") + "." + extension)
            else:
                print("Unknown format.")
                
        else:
            print("The database was not loaded or the model is unknown.")
    
    def corr_visualization(self, columns=None, hue=None, savepath=None):
        """Visualize the correlation between the different parameters/tests.
        
        Arguments:
            - columns: Subset of parameters/tests to consider (passing 'None' considers all the columns).
            - hue: Hue variable (categorical).
            - savepath: Directory where to store the figure (not saved if 'None' is passed).
        """
        # None columns treated as "all" columns
        if columns is None:
            columns = self.db.columns
        
        # Hue variable must be taken into consideration
        if hue not in columns and hue is not None:
            columns = columns + [hue]
            
        # Filename and figure title
        filename = None if savepath is None else (savepath + "/"
                                                  + self._get_model_str(mode="filename") 
                                                  + "_corr_visualization.png")
        title = self._get_model_str(mode="title") + ": Correlation Visualization"
        
        # Plot
        correlation_plot(df=self.db[columns], savepath=filename, title=title, hue=hue)

    def trend(self, columns=None, savepath=None, stack=False, **kwargs):
        """Plot the trend of the given columns
                
        Arguments:
            - columns: Subset of parameters/tests to consider (passing 'None' considers all the columns).
            - savepath: Directory where to store the figure (not saved if 'None' is passed).
            - stack: Boolean. Decide wheter stack all the trends on the same plot.
            - **kwargs: Arguments passed to the 'plot' function.
        """
        if columns is None:
            columns = self.db.columns
            
        if stack and isinstance(self.model, str):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_title(self._get_model_str(mode="title") + ": Trends")
            ax.grid(True)
            ax.set_xlabel("SN")
            for col in columns:
                ax.plot(self.db.index, self.db[col], label=col, **kwargs)
                ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")
        
        elif stack and isinstance(self.model, list):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_title(self._get_model_str(mode="title") + ": Trends")
            ax.grid(True)
            ax.set_xlabel("SN")
            for col in columns:
                for model in self.model:
                    ax.plot(self.db[self.db["Type"]==model].index, self.db[self.db["Type"]==model][col], 
                            label=self._get_model_str(mode="filename") + ": " + col, **kwargs)
                    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")
        
        elif not stack and isinstance(self.model, str):
            nrows = len(columns)
            fig, axs = plt.subplots(nrows=nrows, sharex=False, figsize=(8, 4* nrows), squeeze=False)
            axs[0,0].set_title(self._get_model_str(mode="title") + ": Trends")
            for i, col in enumerate(columns):
                ax = axs[i,0]
                ax.plot(self.db.index, self.db[col], label=col, **kwargs)
                ax.set_title(col)
                ax.grid(True)
            axs[-1,0].set_xlabel("SN")

        elif not stack and isinstance(self.model, list):
            nrows = len(columns)
            fig, axs = plt.subplots(nrows=nrows, sharex=False, figsize=(8, 3.5 * nrows), squeeze=False)
            axs[0,0].set_title(self._get_model_str(mode="title") + ": Trends")
            for i, col in enumerate(columns):
                for model in self.model:
                    ax = axs[i,0]
                    ax.plot(self.db[self.db["Type"]==model].index, self.db[self.db["Type"]==model][col], 
                               label="S" + model, **kwargs)
                    ax.set_ylabel(col)
                    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")
                    ax.grid(True)
            axs[-1,0].set_xlabel("SN")
            
        else:
            print("Model data not loaded")
        
        # Save the plot at given location
        if savepath is not None and "fig" in locals():
            fig.savefig(savepath + "/" + self._get_model_str(mode="filename") + "_trend.png", bbox_inches='tight')  

    def histogram(self, columns=None, which="both", hue=None, savepath=None, **hist_kwags):
        """Plot the histogram and/or the cumulative histogram.
                        
        Arguments:
            - columns: Subset of parameters/tests to consider (passing 'None' considers all the columns).
            - which: Either 'hist' (histogram), 'cum' (cumulative) or 'both'.
            - hue: Hue variable (categorical).
            - savepath: Directory where to store the figure (not saved if 'None' is passed).
            - stack: Boolean. Decide wheter stack all the trends on the same plot.
            - **hist_kwags: Arguments passed to the 'hist' function.
        """
        if columns is None:
            columns = self.db.columns 
        # Hue variable must be taken into consideration
        if hue not in columns and hue is not None:
            columns = columns + [hue]
        
        # Title and filename
        title = self._get_model_str(mode="title")
        filename = savepath + "/" + self._get_model_str(mode="filename") + "_histograms.png" if savepath is not None else None
        
        # Plot
        histogram(self.db[columns], savepath=filename, title=title, which=which, hue=hue,
                  histtype="stepfilled", alpha=0.7, **hist_kwags)
        
    
    def corr_map(self, columns=None, savepath=None):
        """Plot correlation map (heatmap of the correlation matrix.
                                
        Arguments:
            - columns: Subset of parameters/tests to consider (passing 'None' considers all the columns).
            - savepath: Directory where to store the figure (not saved if 'None' is passed).
        """
        if columns is None:
            columns = self.db.columns
        
        # Title and filename
        title = self._get_model_str(mode="title") + ": Correlation Map"
        filename = savepath + "/S" + self._get_model_str(mode="filename") +  "_corr_map.png" if savepath is not None else None
        
        # Plot correlation matrix
        correlation_matrix(self.db[columns].corr(), title=title, clip=True, cbar=False, savepath=filename)
        
class StarlithDb(AbstractDb):
    """Class to load and store all the data of the Starlith mechanisms"""
    
    def __init__(self):
        """Class constructor."""
        super().__init__()
        self.db_motor = None
        self.SN_limits = [None, None]
        
    def load_data(self, model="1900", from_backup=False):
        """Load the Starlith data into a dataframe.
        
        Arguments:
            - model: Either '1400' or '1900' fot the corresponding Starlith model.
            - from_backup: Boolean. Decide whether to load the data from a backup (static) database.
        """
        
        if model == "1900":
            self.model = "1900"
            
            # Directory to search plot
            if from_backup: 
                fetch_dir = ("W:/Organisation/BU_M/3_Development/ME0_Engineering&Development/" +
                             "21. University Projects/DataScience/02_Data/S1900_backup")
            else:
                fetch_dir = ("W:/Projects/W-EM_Starlith_1900_W_NZ530/General/06_Production/" +
                             "01_Archiv_Blenden-_und_Motorendatenbank")
           
            # Manually introduce the position of the cells with the data you want, and the name of this data.
            dataLocation1 = [[6, 1,"SN"], 
                             [35, 7,"Shimmessung"], 
                             [45, 7,"Motormessung"], 
                             [13, 2,"max Kraft"], 
                             [13, 5,"min Kraft"], 
                             [32, 7,"D_Shim_13"], 
                             [33, 7,"D_Shim_11"], 
                             [34, 7,"D_Shim_09"], 
                             [69, 5,"Manual Shift"], 
                             [61, 7,"Hard Stop"], 
                             [51, 5,"LUNA Max: mean+3sigma [mA]"], 
                             [52, 5,"LUNA settling time [s]"], 
                             [53, 5,"LUNA Max: max [mA]"], 
                             [32, 3,"Shiftx_13"], 
                             [33, 3,"Shiftx_11"], 
                             [34, 3,"Shiftx_09"], 
                             [32, 4,"Shifty_13"], 
                             [33, 4,"Shifty_11"], 
                             [34, 4,"Shifty_09"], 
                             [32, 5,"Ellipse_13"], 
                             [33, 5,"Ellipse_11"], 
                             [34, 5,"Ellipse_09"], 
                             [32, 6,"dz_13"], 
                             [33, 6,"dz_11"], 
                             [34, 6,"dz_09"], 
                             [42, 7,"D_Motor_13"], 
                             [43, 7,"D_Motor_11"],
                             [44, 7,"D_Motor_09"], 
                             [8, 5, "Motor SN"], 
                             [8, 5, "Motor Lot"],
                             [76, 5, "Spalt max 1-2"],
                             [77, 5, "Spalt max 2-3"],
                             [78, 5, "Spalt max 3-4"],
                             [79, 5, "Spalt max 4-5"],
                             [80, 5, "Spalt max 5-6"],
                             [81, 5, "Spalt max 6-7"],
                             [82, 5, "Spalt max 7-8"],
                             [83, 5, "Spalt max 8-1"],
                             [76, 7, "Spalt min 1-2"],
                             [77, 7, "Spalt min 2-3"],
                             [78, 7, "Spalt min 3-4"],
                             [79, 7, "Spalt min 4-5"],
                             [80, 7, "Spalt min 5-6"],
                             [81, 7, "Spalt min 6-7"],
                             [82, 7, "Spalt min 7-8"],
                             [83, 7, "Spalt min 8-1"]]
            
            dataLocation2 = [[6, 1, "SN"], 
                             [35, 8, "Shimmessung"], 
                             [62, 3, "Motormessung"], 
                             [41, 2, "max Kraft"], 
                             [41, 5, "min Kraft"], 
                             [32, 8, "D_Shim_13"], 
                             [33, 8, "D_Shim_11"], 
                             [34, 8, "D_Shim_09"], 
                             [59, 7, "Manual Shift"], 
                             [69, 7, "Hard Stop"], 
                             [50, 5, "LUNA Max: mean+3sigma [mA]"], 
                             [51, 5, "LUNA settling time [s]"], 
                             [52, 5, "LUNA Max: max [mA]"],
                             [32, 3, "Shiftx_13"], 
                             [33, 3, "Shiftx_11"], 
                             [34, 3, "Shiftx_09"], 
                             [32, 4, "Shifty_13"], 
                             [33, 4, "Shifty_11"], 
                             [34, 4, "Shifty_09"],
                             [32, 6, "Ellipse_13"], 
                             [33, 6, "Ellipse_11"], 
                             [34, 6, "Ellipse_09"], 
                             [32, 7, "dz_13"], 
                             [33, 7, "dz_11"], 
                             [34, 7, "dz_09"],
                             [59, 4, "D_Motor_13"], 
                             [60, 4, "D_Motor_11"], 
                             [61, 4, "D_Motor_09"], 
                             [8, 5, "Motor SN"], 
                             [8, 7, "Motor Lot"],
                             [12, 5, "Spalt max 1-2"],
                             [13, 5, "Spalt max 2-3"],
                             [14, 5, "Spalt max 3-4"],
                             [15, 5, "Spalt max 4-5"],
                             [16, 5, "Spalt max 5-6"],
                             [17, 5, "Spalt max 6-7"],
                             [18, 5, "Spalt max 7-8"],
                             [19, 5, "Spalt max 8-1"],
                             [12, 7, "Spalt min 1-2"],
                             [13, 7, "Spalt min 2-3"],
                             [14, 7, "Spalt min 3-4"],
                             [15, 7, "Spalt min 4-5"],
                             [16, 7, "Spalt min 5-6"],
                             [17, 7, "Spalt min 6-7"],
                             [18, 7, "Spalt min 7-8"],
                             [19, 7, "Spalt min 8-1"]]
        
        elif model=="1400":
            self.model = "1400"
            
            # Directory to search pplot
            if from_backup:
                fetch_dir = ("W:/Organisation/BU_M/3_Development/ME0_Engineering&Development/" +
                             "21. University Projects/DataScience/02_Data/S1400_backup")
            else:
                fetch_dir = ("W:/Projects/W-EM_Starlith_1400_W_NZ365/General/06_Production/" +
                             "Archiv_Blendendatenbanken")

            # Manually introduce the position of the cells with the data you want, and the name of this data.
            dataLocation1 = [[6, 1, "SN"], 
                             [35, 7, "Shimmessung"], 
                             [45, 7, "Motormessung"], 
                             [13, 2, "max Kraft"], 
                             [13, 5, "min Kraft"], 
                             [32, 7, "D_Shim_13"], 
                             [33, 7, "D_Shim_11"], 
                             [34, 7, "D_Shim_09"], 
                             [51, 5, "LUNA Max: mean+3sigma [mA]"], 
                             [52, 5, "LUNA settling time [s]"],
                             [53, 5, "LUNA Max: max [mA]"], 
                             [32, 3, "Shiftx_13"], 
                             [33, 3, "Shiftx_11"], 
                             [34, 3, "Shiftx_09"], 
                             [32, 4, "Shifty_13"], 
                             [33, 4, "Shifty_11"], 
                             [34, 4, "Shifty_09"],
                             [32, 5, "Ellipse_13"], 
                             [33, 5, "Ellipse_11"], 
                             [34, 5, "Ellipse_09"], 
                             [32, 6, "dz_13"], 
                             [33, 6, "dz_11"],
                             [34, 6, "dz_09"],
                             [42, 7, "D_Motor_13"], 
                             [43, 7, "D_Motor_11"],
                             [44, 7, "D_Motor_09"], 
                             [8, 5, "Motor SN"], 
                             [8, 5, "Motor Lot"],
                             [76, 5, "Spalt max 1-2"],
                             [77, 5, "Spalt max 2-3"],
                             [78, 5, "Spalt max 3-4"],
                             [79, 5, "Spalt max 4-5"],
                             [80, 5, "Spalt max 5-6"],
                             [81, 5, "Spalt max 6-7"],
                             [82, 5, "Spalt max 7-8"],
                             [83, 5, "Spalt max 8-1"],
                             [76, 7, "Spalt min 1-2"],
                             [77, 7, "Spalt min 2-3"],
                             [78, 7, "Spalt min 3-4"],
                             [79, 7, "Spalt min 4-5"],
                             [80, 7, "Spalt min 5-6"],
                             [81, 7, "Spalt min 6-7"],
                             [82, 7, "Spalt min 7-8"],
                             [83, 7, "Spalt min 8-1"]]
            
            dataLocation2 = [[6, 1, "SN"], 
                             [35, 8, "Shimmessung"], 
                             [62, 3, "Motormessung"], 
                             [41, 2, "max Kraft"], 
                             [41, 5, "min Kraft"], 
                             [32, 8, "D_Shim_13"], 
                             [33, 8, "D_Shim_11"],
                             [34, 8, "D_Shim_09"], 
                             [50, 5, "LUNA Max: mean+3sigma [mA]"],
                             [51, 5, "LUNA settling time [s]"], 
                             [52, 5, "LUNA Max: max [mA]"], 
                             [32, 3, "Shiftx_13"], 
                             [33, 3, "Shiftx_11"], 
                             [34, 3, "Shiftx_09"], 
                             [32, 4, "Shifty_13"], 
                             [33, 4, "Shifty_11"],
                             [34, 4, "Shifty_09"],
                             [32, 6, "Ellipse_13"], 
                             [33, 6, "Ellipse_11"], 
                             [34, 6, "Ellipse_09"], 
                             [32, 7, "dz_13"],
                             [33, 7, "dz_11"], 
                             [34, 7, "dz_09"], 
                             [59, 4, "D_Motor_13"], 
                             [60, 4, "D_Motor_11"],
                             [61, 4, "D_Motor_09"], 
                             [8, 5, "Motor SN"], 
                             [8, 7, "Motor Lot"],
                             [12, 5, "Spalt max 1-2"],
                             [13, 5, "Spalt max 2-3"],
                             [14, 5, "Spalt max 3-4"],
                             [15, 5, "Spalt max 4-5"],
                             [16, 5, "Spalt max 5-6"],
                             [17, 5, "Spalt max 6-7"],
                             [18, 5, "Spalt max 7-8"],
                             [19, 5, "Spalt max 8-1"],
                             [12, 7, "Spalt min 1-2"],
                             [13, 7, "Spalt min 2-3"],
                             [14, 7, "Spalt min 3-4"],
                             [15, 7, "Spalt min 4-5"],
                             [16, 7, "Spalt min 5-6"],
                             [17, 7, "Spalt min 6-7"],
                             [18, 7, "Spalt min 7-8"],
                             [19, 7, "Spalt min 8-1"]]
    
        else:
            raise ValueError("Unknown model '{}'".format(model))
            
        # Fetch all files in fetch_dir
        full_file_list = []
        for file in os.listdir(fetch_dir):
            if file[:16] == "Blendendatenbank":
                 full_file_list.append(fetch_dir + "/" + file)

        # Sorting files according to increasing SN.
        sorted_files, SN_range_sorted = self._sort_files(full_file_list)
        self.files = sorted_files.copy()
        self.SN_limits = [SN_range_sorted[0][0], SN_range_sorted[-1][1]]
        

        #Initialize data frames
        df = pd.DataFrame()
        dfLine=pd.DataFrame()

        #Read every file
        print("Loading data...")
        count = 0
        for f in self.files:
            workbook = xlrd.open_workbook(f)
            for s in workbook.sheets():
                if s.name[0:2]=="SN" and len(s.name) <= 6:
                    SN = int(s.name[2:])
                    
                    # Display progress
                    count = count + 1
                    print(f"\rProgress: {count}/{self.SN_limits[1]-self.SN_limits[0] + 1}  ", end = "") 
                    
                    # Check which template the data is in (1=old, 2=new)
                    if (s.cell_value(11,4) == "Spalt bei max NA: [mm]" 
                        and s.cell_value(21,10) == "Path of the folder with the files"):
                        T = "Template2"
                        dfLine = pd.DataFrame()
                        #Get every piece of data and put it in the data frame
                        for set in dataLocation2:
                            row = set[0]
                            col = set[1]
                            desc = set[2]
                            value = s.cell_value(row, col)
                        
                            #Consider as no value the values = 0 that can come from adding two empty cells
                            if value == 0:
                                value = "no value"
                                
                            # Sometimes cells are in text mode even though they are number, 
                            # convert them to floats
                            try:
                                if desc == "Motor Lot":
                                    value = int(value)
                                elif desc == "Motor SN":
                                    value = int(value.replace("SN", ""))
                                else:
                                    value=float(value)
                            except:
                                continue
                            
                            dfNew = pd.DataFrame(index=[SN], columns=[desc],
                                                 data=[[value]] if (isinstance(value, float) or 
                                                                    isinstance(value, int)) else np.nan)
                            dfLine = pd.concat([dfNew, dfLine], sort=True, axis=1)
                        
                        #Number of shim measurements
                        proxynshim=s.cell_value(34,10)
                        l1=len(proxynshim)
                        if l1!=0:
                            if proxynshim[(l1-6)] =="0":                         
                                dfnshim = pd.DataFrame(index=[SN], columns=["nshim"], 
                                                       data=[[int(proxynshim[(l1-5)])]])
                                
                            elif proxynshim[(l1-6)]=="1" or proxynshim[(l1-6)]=="2":
                                dfnshim = pd.DataFrame(index=[SN], columns=["nshim"], 
                                                       data=[[int(proxynshim[(l1-6):(l1-4)])]])
                                
                            else:
                                #for when files were called differently, first units usually
                                dfnshim = pd.DataFrame(index=[SN], columns=["nshim"],
                                                           data=[[int(proxynshim[(l1-5)])]])                        
                        else:
                            #if no file was put, we assume at least 1 measurement
                            dfnshim=pd.DataFrame(index=[SN], columns=["nshim"],data=[[1]])
                        dfLine= pd.concat([dfnshim,dfLine], sort=True, axis=1)

                       #Number of motor measurements
                        proxynmotor=s.cell_value(61,10)
                        l2=len(proxynmotor)
                        if l2!=0:
                            if proxynmotor[(l2-6)] =="0":                         
                                dfnmotor=pd.DataFrame(index=[SN], columns=["nmotor"],
                                                      data=[[int(proxynmotor[(l1-5)])]])
                                dfnmotor=pd.DataFrame(index=[SN], columns=["nmotor"],
                                                      data=[[int(proxynmotor[(l2-6):(l2-4)])]])
                            else:
                                #for when files were called differently, first units usually
                                dfnmotor=pd.DataFrame(index=[SN], columns=["nmotor"],
                                                      data=[[int(proxynmotor[(l2-5)])]])                        
                        else:
                            #if no file was put, we assume at least 1 measurement
                            dfnmotor=pd.DataFrame(index=[SN], columns=["nmotor"], data=[[1]])
                        dfLine= pd.concat([dfnmotor,dfLine], sort=True, axis=1)
                        
                        #Add template number
                        dfTemplate=pd.DataFrame(index=[SN], columns=["Template"], data=[[T]])
                        dfLine= pd.concat([dfTemplate,dfLine], sort=True, axis=1)
                        
                        df= pd.concat([dfLine,df], sort=True, axis=0)
                        
                    elif s.cell_value(86,1)== "Teile gesichert":
                        T="Template1"
                        dfLine = pd.DataFrame()
                        #Get every piece of data and put it in the data frame
                        for set in dataLocation1:
                            row = set[0]
                            col = set[1]
                            desc = set[2]
                            value=s.cell_value(row, col)
                            
                            # Consider as no value the values = 0 that can come from 
                            # adding two empty cells
                            
                            if value == 0:
                                value = "no value"
                            
                            # Sometimes cells are in text mode even though they are number, 
                            # convert them to floats
                            try:
                                if desc == "Motor Lot":
                                    value = int(value.split("SN")[0])
                                elif desc == "Motor SN":
                                    value = int(value.split("SN")[1])
                                else:
                                    value=float(value)
                            except:
                                continue
                                
                            dfNew = pd.DataFrame(index=[SN], columns=[desc],
                                                 data=[[value]] if (isinstance(value, float) 
                                                                    or isinstance(value, int)) else np.nan)
                            dfLine = pd.concat([dfNew, dfLine], sort=True, axis=1)
                        
                        #Number of shim measurements
                        proxynshim=s.cell_value(34,9)
                        l1=len(proxynshim)
                        if l1!=0:
                            if proxynshim[(l1-6)] =="0":                         
                                dfnshim=pd.DataFrame(index=[SN], columns=["nshim"],
                                                         data=[[int(proxynshim[(l1-5)])]])
                            elif proxynshim[(l1-6)]=="1" or proxynshim[(l1-6)]=="2":
                                dfnshim=pd.DataFrame(index=[SN], columns=["nshim"],
                                                     data=[[int(proxynshim[(l1-6):(l1-4)])]])
                            else:
                                #for when files were called differently, first units usually
                                dfnshim=pd.DataFrame(index=[SN], columns=["nshim"],
                                                     data=[[int(proxynshim[(l1-5)])]])                        
                        else:
                            #if no file was put, we assume at least 1 measurement
                            dfnshim=pd.DataFrame(index=[SN], columns=["nshim"], data=[[1]])
                        dfLine= pd.concat([dfnshim,dfLine], sort=True, axis=1)
                            

                        #Number of motor measurements
                        proxynmotor=s.cell_value(44,9)
                        l2=len(proxynmotor)
                        if l2!=0:
                            if proxynmotor[(l2-6)] =="0":                         
                                dfnmotor=pd.DataFrame(index=[SN], columns=["nmotor"],
                                                      data=[[int(proxynmotor[(l1-5)])]])
                            elif proxynmotor[(l2-6)]=="1" or proxynmotor[(l2-6)]=="2":
                                dfnmotor=pd.DataFrame(index=[SN], columns=["nmotor"],
                                                      data=[[int(proxynmotor[(l2-6):(l2-4)])]])
                            else:
                                #for when files were called differently, first units usually
                                dfnmotor=pd.DataFrame(index=[SN], columns=["nmotor"],
                                                      data=[[int(proxynmotor[(l2-5)])]])                        
                        else:
                            #if no file was put, we assume at least 1 measurement
                            dfnmotor=pd.DataFrame(index=[SN], columns=["nmotor"],data=[[1]])
                        dfLine= pd.concat([dfnmotor,dfLine], sort=True, axis=1)
                        
                        #Add template number
                        dfTemplate=pd.DataFrame(index=[SN], columns=["Template"],data=[[T]])
                        dfLine= pd.concat([dfTemplate, dfLine], sort=True, axis=1)
                        
                        df = pd.concat([dfLine, df], sort=True, axis=0)
                        
                    else:
                        T="Unknown Template"
                        dfTemplate=pd.DataFrame(index=[SN], columns=["Template"],data=[[T]])
                        dfLine= pd.concat([dfTemplate, dfLine], sort=True, axis=1)
                        df= pd.concat([dfLine, df], sort=True, axis=0)
        
        # SN already in index
        df.drop(columns="SN", inplace=True)

        #Extra computations
        df["MM-SM"]=df.Motormessung-df.Shimmessung
        df["Dezentrierung13"]=(df.Shiftx_13**2+df.Shifty_13**2)**0.5
        df["Dezentrierung11"]=(df.Shiftx_11**2+df.Shifty_11**2)**0.5
        df["Dezentrierung09"]=(df.Shiftx_09**2+df.Shifty_09**2)**0.5 
        df["extra cost"] =  (df.nmotor.where(df.nmotor == 1, 2) - 1)
        
        # Store model information
        df["Type"] = self.model
        
        # Laod motor data
        print("\nDone.\nLoading motor data...")
        self._load_motor_data(from_backup=from_backup)
        print("Done.")
        
        # Join motor and mechanism data
        df["PISEL Up [mA]"] = np.nan
        df["PISEL Down [mA]"] = np.nan

        for SN in df.index:
            motor_SN = df["Motor SN"].loc[SN]
            motor_Lot = df["Motor Lot"].loc[SN]
            df_tmp = self.db_motor[self.db_motor["RUAG Los Number"] == motor_Lot]
            if motor_SN in df_tmp.index:
                df.at[SN, "PISEL Up [mA]"] = df_tmp["Stromverbrauch (+/-12VDC) [mA] Up"].loc[motor_SN].astype(float)
                df.at[SN, "PISEL Down [mA]"] = df_tmp["Stromverbrauch (+/-12VDC) [mA] Down"].loc[motor_SN].astype(float)
                    
        
        # Store database
        self.db_original = df.copy()
        self.db = df.copy()

    
    def join(db1, db2):
        """Join two databases.
        
        Arguments:
            - db1: First database.
            - db2: Second database.
        Return:
            - A merged database object.
        """
        # Extract and copy data
        df1_original = db1.db_original.copy()
        df2_original = db2.db_original.copy()
        df1 = db1.db.copy()
        df2 = db2.db.copy()
        
        # Merge database
        df_merged = pd.concat([df1, df2], sort=False).drop_duplicates()
        df_merged_original = pd.concat([df1_original, df2_original], sort=False).drop_duplicates()
        
        # Create new db
        db_merged = StarlithDb()
        db_merged.model = [db1.model, db2.model] if db1.model != db2.model else db1.model
        db_merged.db_original = df_merged_original.copy()
        db_merged.db = df_merged.copy()
        
        return db_merged
        
    def _load_motor_data(self, from_backup=False):
        """[Private] Load motor database.
        
        Arguments:
            - from_backup: Boolean. Decide whether to load the data from a backup (static) database.
        """
        
        # Load motor data
        if from_backup:
            motor_db_path = ("W:/Organisation/BU_M/3_Development/ME0_Engineering&Development/" + 
                             "21. University Projects/DataScience/02_Data/Motoren_Datenbank.xlsx")
        else:
            motor_db_path= "W:/Projects/W-EM_Starlith_1900_W_NZ530/General/06_Production/Motoren_Datenbank.xlsx"

        # Load database
        df_motor = pd.read_excel(motor_db_path, sheet_name="Alle Motoren", 
                                 header=7, usecols=[1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        
        
        # Rename columns
        df_motor.rename(columns={old_name : old_name.replace("\n", " ") 
                                 for old_name in df_motor.columns}, inplace=True)
        # Drop incompleter models
        df_motor.dropna(subset=["Stromverbrauch (+/-12VDC) [mA] Up", 
                                "Stromverbrauch (+/-12VDC) [mA] Down", 
                                "RUAG Serie Nummer"], inplace=True)
        
        # Rename index
        df_motor = df_motor.astype({"RUAG Los Number" : int })
        df_motor["RUAG Serie Nummer"] = df_motor["RUAG Serie Nummer"].map(lambda x : int(x.replace("SN", "").replace(".", "").split("(")[0]))
        df_motor.drop_duplicates(subset=["RUAG Serie Nummer", "RUAG Los Number"], inplace=True)
        df_motor.set_index("RUAG Serie Nummer", inplace=True)
        df_motor.sort_index(inplace=True)
        
        # Store database in class instance
        self.db_motor = df_motor
                
            
    def _sort_files(self, files_list):
        """[Private] Sort a list of files according to their SN.
        Arguments:
            - files_list: List of files to sort.
        Return:
            - A sorted list of files.
        """
        
        SN_range_list = [tuple(map(int, file.split("_")[-1].split(".")[0].split("-"))) for file in files_list]
        sorted_files = [file for file, _ in sorted(zip(files_list, SN_range_list), key=lambda x : x[1][0])]
        SN_range_sorted = sorted(SN_range_list, key=lambda x : x[0])
        return sorted_files, SN_range_sorted
    
    def _get_model_str(self, mode="filename"):
        """[Private] Return a string of the model(s) name (for plot titles, filenames, etc.)
        
        Arguments:
            -mode: Either 'title' or 'filename' (short and without space).
        """
        string = ""
        if self.model is None:
            string = "None"
        elif isinstance(self.model, list) and mode == "title":
            string = "Starlith " + " & ".join(self.model)
        elif isinstance(self.model, list) and mode == "filename":
            string = "S" + "&".join(self.model)
        elif isinstance(self.model, str) and mode == "title":
            string = "Stalith " + self.model
        elif isinstance(self.model, str) and mode == "filename":
            string = "S" + self.model
        
        return string
    
    def generate_plots(self, path="./plots/"):
        raise NotImplementedError
    
###########################################
############        ECD        ############
###########################################

class ECDDb(AbstractDb):
    """Databse class for ECD data"""
    
    def __init__(self):
        """Class constructor."""
        super().__init__()
        self.raw_data_dict = None
        self.processed_data_dict = None
        self.fft_data_dict = None
        self.gear_ratio = 2548
        self.specifications = {"ST [Nm]": 1.24,
                               "DR min [Nms/rad]": 500,
                               "DR max [Nms/rad]": 1700,
                               "Repeatability [%]": 10,
                               "Reversability [%]": 5,
                               "Frequency Shift [%]": 10,
                               "Play axial [mm]": 0.1,
                               "Play radial [mm]": 0.25}
    
    def load_data(self, db_dir, update=False, raw=False):
        """Load ECD data.
        
        Arguments:
            - db_dir: Path to the directory where the database and the metadata is stored (or should be stored).
            - update: Boolean. Decide wheter the database should be updated.
            - raw: Boolean. Decide if the raw data (csv files) should be loaded.
            """
        
        self.model = "ECD"
        self.db_dir = db_dir
        # Get raw files path
        self.raw_files = pd.read_excel(db_dir + "/MetaData.xlsx", "Filepaths",
                                       index_col=0).drop(columns=["Summary", "Template"]).sort_index()
        self.step_names = list(self.raw_files.columns)
        
        # Load summary files
        if update:
            # Get summary file paths and template
            self.files = pd.read_excel(db_dir + "/MetaData.xlsx", "Filepaths", index_col=0)[["Summary", "Template"]]
            self.files.sort_index(inplace=True)
            self.templates = pd.read_excel(db_dir + "/MetaData.xlsx", sheet_name="Templates",index_col=0, header = [0,1])
            
            # Initialize df
            df = pd.DataFrame(index=self.files.index, columns=list(self.templates.columns.unique(level=0)))
            
            # Loading the summary files
            count = 0
            for FM, file, template in zip(self.files.index, self.files["Summary"], self.files["Template"]):
        
                # Open sheet
                workbook = xlrd.open_workbook(file)
                sheet = workbook.sheets()[0]
                dict_tmp = {}

                # Load cell by cell
                for name in df.columns:
                    row = self.templates.loc[template, (name, "row")]
                    col = self.templates.loc[template, (name, "col")]
                    value = sheet.cell_value(row, col)
                    if value == "":
                        value = np.nan
                    else:
                        try:
                            value = float(value)
                        except:
                            print("FM{}: Wrong data format ({}): ".format(FM, name), value)

                    dict_tmp[name] = value
                df.loc[FM] = dict_tmp
                count += 1
                print("\r{}/{} files loaded".format(count, len(self.files)), end="   ")

            # Storing the db in the object instance
            df.sort_index(inplace=True)
            self.db_original = df

            # Save database
            self.save_db(self.db_dir, extension="csv")
            
        else:
            df = pd.read_csv(self.db_dir + "/DataBaseExport_ECD.csv", index_col=0)
            self.db_original = df
            print("Existing database loaded.")
        
        # Compute secondary value
        self._compute_secondary_values()
        
        # Load raw data (from csv files)
        if raw:       
            # Initializing the raw data dictionary
            self.raw_data_dict = {FM : {test : None for test in self.step_names} for FM in self.raw_files.index}
            
            # Loading every file
            total_memory = 0
            for FM in self.raw_files.index:
                print("\rLoading the raw data (FM {})".format(FM), end="  ")
                for test in self.step_names:
                    filepath_CH = self.raw_files.at[FM, test]
                    # Avoid loading absent data
                    if str(filepath_CH) != 'nan':
                        # Exctract channel and file path for the (FM, test) combination
                        channel = int(self.raw_files.at[FM, test].split(" CH")[-1])
                        filepath = self.raw_files.at[FM, test].split(" CH")[0]
                        # Read the csv (txt with ; seperator) file
                        df = pd.read_csv(filepath, sep=";", header=4, skip_blank_lines=False, 
                                         parse_dates=True, encoding = 'unicode_escape')
                        # Modify the index
                        df["Datetime [y-m-d h:m:s]"] = pd.to_datetime(df["Date[d.m.y]"] + " " + df["Exact Time[h.min.s.ms]"])
                        df.drop(columns=["Date[d.m.y]", "Exact Time[h.min.s.ms]"], inplace=True)
                        df.set_index("Datetime [y-m-d h:m:s]", inplace=True)
                        # Drop useless channels and rename the good one
                        df.drop(columns=["CH{}[Nm] ()".format(c) for c in [1, 2, 3, 4] if c != channel], inplace=True)
                        df.rename(columns={"CH{}[Nm] ()".format(channel): "Data [Nm]"}, inplace=True)
                        # Drop columns and rows filled with nan values
                        df.dropna(axis=0, how="all", inplace=True)
                        df.dropna(axis=1, how="all", inplace=True)
                        # Reset elapsed time so it starts at 0 second
                        df['Elapsed Time[s]'] = df['Elapsed Time[s]'] - df['Elapsed Time[s]'].iloc[0]
                        # Compute data usage and store data in the class instance
                        total_memory += df.memory_usage().sum() / 1e6 
                        self.raw_data_dict[FM][test] = df
            
            # Process raw data for AI
            self._process_raw_data()
            
            print("\nDone (Total memory: {} Mbytes)".format(total_memory))
        
        # Copy the original data to keep an untouched version
        self.db = self.db_original.copy()
        
    def _get_model_str(self, mode):
        """[Private] Return the string indicating what model is loaded into this database objcet."""
        return self.model
    
    def _process_raw_data(self):
        """Process the raw data in the raw_data_dict and create a new dictionary processed_data_dict with the processed data."""
        self.processed_data_dict = {}
        self.fft_data_dict = {}
        for FM, FM_dict in self.raw_data_dict.items():
            self.processed_data_dict[FM] = {}
            self.fft_data_dict[FM] = {}
            isComplete = True
            for test in FM_dict.keys():
                if FM_dict[test] is None:
                    isComplete = False
                elif "ST" in test:
                    df = FM_dict[test]
                    
                    # Backlash 
                    backlash_data = df[df["Inner Phase"] == "Ramp 2nd Speed"]["Data [Nm]"]
                    backlash = backlash_data.iloc[len(backlash_data) // 2:-1].mean()
                    
                    # Test for the sampling ratio
                    test_dr = test.replace("ST", "DR").replace("[Nm]", "[Nms/rad]")
                    
                    # Start and end index to consider in the `Hold 2nd Speed` inner phase
                    idx_start_ST = 0
                    idx_end_ST = 30500
                    
                    #RPM (before gearbox)
                    speed = 10 
                    
                    # Extract the data
                    data = np.abs(df[df["Inner Phase"] == "Hold 2nd Speed"]["Data [Nm]"].sub(backlash)).iloc[idx_start_ST: idx_end_ST]
                    data = data - 0.95 * self.db_original.at[FM, test_dr] * 2 * np.pi * speed / (60 * self.gear_ratio)
                    
                    if len(data) != idx_end_ST - idx_start_ST:
                        print("Careful, the test " + test + " for FM" + FM + " has an unexpected size ({} instead of {})".format(idx_end_ST - idx_start_ST, len(data)))
                        
                    # Compute fft
                    data_fft = fft(data)
                    
                    # Insert the data in the dictionaries
                    self.processed_data_dict[FM][test] = data.values
                    self.fft_data_dict[FM][test] = data_fft
            
            # Delete incomplete sample (no suited for learning)
            if not isComplete:
                del self.processed_data_dict[FM]
    
    def _compute_secondary_values(self):
        """Compute the secondary values that are also constrained by specifications 
        (e.g. reversability, frequency shift, etc).
        """
        
        # Reversability or damping rates
        self.db_original["DR Revers. (< Strength) [%]"]  = np.abs(self.db_original["DR CW (< Strength) [Nms/rad]"].div(self.db_original["DR CCW (< Strength) [Nms/rad]"]) * 100 - 100)
        self.db_original["DR Revers. (> Strength) [%]"]  = np.abs(self.db_original["DR CW (> Strength) [Nms/rad]"].div(self.db_original["DR CCW (> Strength) [Nms/rad]"]) * 100 - 100)
        self.db_original["DR Revers. (> Vibration) [%]"] = np.abs(self.db_original["DR CW (> Vibration) [Nms/rad]"].div(self.db_original["DR CCW (> Vibration) [Nms/rad]"]) * 100 - 100)
        self.db_original["DR Revers. (1st +22°C) [%]"]   = np.abs(self.db_original["DR CW (1st +22°C) [Nms/rad]"].div(self.db_original["DR CCW (1st +22°C) [Nms/rad]"]) * 100 - 100)
        self.db_original["DR Revers. (1st +100°C) [%]"]  = np.abs(self.db_original["DR CW (1st +100°C) [Nms/rad]"].div(self.db_original["DR CCW (1st +100°C) [Nms/rad]"]) * 100 - 100)
        self.db_original["DR Revers. (1st -55°C) [%]"]   = np.abs(self.db_original["DR CW (1st -55°C) [Nms/rad]"].div(self.db_original["DR CCW (1st -55°C) [Nms/rad]"]) * 100 - 100)
        self.db_original["DR Revers. (4th +100°C) [%]"]  = np.abs(self.db_original["DR CW (4th +100°C) [Nms/rad]"].div(self.db_original["DR CCW (4th +100°C) [Nms/rad]"]) * 100 - 100)
        self.db_original["DR Revers. (4th -55°C) [%]"]   = np.abs(self.db_original["DR CW (4th -55°C) [Nms/rad]"].div(self.db_original["DR CCW (4th -55°C) [Nms/rad]"]) * 100 - 100)
        self.db_original["DR Revers. (4th +22°C) [%]"]   = np.abs(self.db_original["DR CW (4th +22°C) [Nms/rad]"].div(self.db_original["DR CCW (4th +22°C) [Nms/rad]"]) * 100 - 100)
        
        #  DR Repeatability CW
        self.db_original["DR CW Repeat. (< Strength to > Strength) [%]"]  = np.abs(self.db_original["DR CW (> Strength) [Nms/rad]"].div(self.db_original["DR CW (< Strength) [Nms/rad]"])  * 100 - 100)
        self.db_original["DR CW Repeat. (> Strength to > Vibration) [%]"] = np.abs(self.db_original["DR CW (> Vibration) [Nms/rad]"].div(self.db_original["DR CW (> Strength) [Nms/rad]"]) * 100 - 100)
        self.db_original["DR CW Repeat. (1st +100°C to 4th +100°C) [%]"]  = np.abs(self.db_original["DR CW (4th +100°C) [Nms/rad]"].div(self.db_original["DR CW (1st +100°C) [Nms/rad]"])  * 100 - 100)
        self.db_original["DR CW Repeat. (1st -55°C to 4th -55°C) [%]"]    = np.abs(self.db_original["DR CW (4th -55°C) [Nms/rad]"].div(self.db_original["DR CW (1st -55°C) [Nms/rad]"])   * 100 - 100)
        self.db_original["DR CW Repeat. (1st +22°C to 4th +22°C) [%]"]    = np.abs(self.db_original["DR CW (4th +22°C) [Nms/rad]"].div(self.db_original["DR CW (1st +22°C) [Nms/rad]"])    * 100 - 100)

        #  DR Repeatability CCW
        self.db_original["DR CCW Repeat. (< Strength to > Strength) [%]"]  = np.abs(self.db_original["DR CCW (> Strength) [Nms/rad]"].div(self.db_original["DR CCW (< Strength) [Nms/rad]"])  * 100 - 100)
        self.db_original["DR CCW Repeat. (> Strength to > Vibration) [%]"] = np.abs(self.db_original["DR CCW (> Vibration) [Nms/rad]"].div(self.db_original["DR CCW (> Strength) [Nms/rad]"]) * 100 - 100)
        self.db_original["DR CCW Repeat. (1st +100°C to 4th +100°C) [%]"]  = np.abs(self.db_original["DR CCW (4th +100°C) [Nms/rad]"].div(self.db_original["DR CCW (1st +100°C) [Nms/rad]"])  * 100 - 100)
        self.db_original["DR CCW Repeat. (1st -55°C to 4th -55°C) [%]"]    = np.abs(self.db_original["DR CCW (4th -55°C) [Nms/rad]"].div(self.db_original["DR CCW (1st -55°C) [Nms/rad]"])   * 100 - 100)
        self.db_original["DR CCW Repeat. (1st +22°C to 4th +22°C) [%]"]    = np.abs(self.db_original["DR CCW (4th +22°C) [Nms/rad]"].div(self.db_original["DR CCW (1st +22°C) [Nms/rad]"])    * 100 - 100)
        
        # Frequency shft
        self.db_original["Frequency Shift Z [%]"]  = np.abs(self.db_original["Resonance Z (> Random) [Hz]"].div(self.db_original["Resonance Z (< Random) [Hz]"])  * 100 - 100)
        self.db_original["Frequency Shift X [%]"]  = np.abs(self.db_original["Resonance X (> Random) [Hz]"].div(self.db_original["Resonance X (< Random) [Hz]"])  * 100 - 100)
     
    
    
    
    def visualize_raw_data(self, FM, test, savepath=None, start_time=None, end_time=None):
        """Visualize the raw data that is used to compute the Stiction torque or the Damping ratio.
        
        Arguments:
            - FM: Model to visualize.
            - test: The particular test to visualize.
            - savepath: Directory where to store the plots.
            - start_time: Start of the interval used to compute the given value (pass 'None' to infer automatically).
            - end_time: End of the interval (pass 'None' to infer automatically).
        """
        # Extract data
        df = self.raw_data_dict[FM][test]
        
        # Stiction torque test
        if "ST" in test:
            # Set relevant parameters / tests
            test_max = test.replace("W (", "W Max (")
            test_dr = test.replace("ST", "DR").replace("[Nm]", "[Nms/rad]")
            if start_time is None:
                start_time = 1000
            if end_time is None:
                end_time = 15500
            
            # Infer backlash
            backlash_data = df[df["Inner Phase"] == "Ramp 2nd Speed"]["Data [Nm]"]
            backlash = backlash_data.iloc[len(backlash_data) // 2:-1].mean()
        
            max_torque = max(np.abs(df["Data [Nm]"].loc[(df["Elapsed Time[s]"] > start_time) 
                                                        & (df["Elapsed Time[s]"] < end_time)].sub(backlash).max()),
                             np.abs(df["Data [Nm]"].loc[(df["Elapsed Time[s]"] > start_time) 
                                            & (df["Elapsed Time[s]"] < end_time)].sub(backlash).min()))
            
            ST = max_torque - 0.95 * self.db.at[FM, test_dr] * 2 * np.pi * 10 / (60 * self.gear_ratio)
            
            description =("Actual ST: {:.2f} [Nm]".format(ST) +
                          "\nActual ST (Summary Sheet): {:.2f} [Nm]".format(self.db.at[FM, test]) +
                          "\n\nParameters:" +
                          "\n-Backlash: {:.2f}".format(backlash) +
                          "\n-Start time: {:n}".format(start_time) +
                          "\n-End time: {:n}".format(end_time))
    
        #Damping rate test
        elif "DR" in test:
            if start_time is None:
                start_time = df[df["Inner Phase"] == "Hold 3rd Speed"]["Elapsed Time[s]"].iloc[0]
            if end_time is None:
                end_time = df[df["Inner Phase"] == "Hold 3rd Speed"]["Elapsed Time[s]"].iloc[-1]
                
            rpm = int(df[df["Inner Phase"] == "Hold 3rd Speed"]["Speed"].iloc[0][1:])
            DR = df["Data [Nm]"].loc[(df["Elapsed Time[s]"] > start_time) 
                                     & (df["Elapsed Time[s]"] < end_time)].mean() * 60 * self.gear_ratio / (2 * np.pi * rpm)
            
            description =("Actual DR: {:.2f} [Nms/rad]".format(DR) +
                          "\nActual DR (Summary Sheet): {:.2f} [Nms/rad]".format(self.db.at[FM, test]) +
                          "\n\nParameters:" +
                          "\n-Start time: {:n}".format(start_time) +
                          "\n-End time: {:n}".format(end_time))
            
        else:
            raise ValueError("Unknown test.")
        
        # Visualization
        fig, ax = plt.subplots(1, 1, figsize=(16,4))
        ax.plot(df["Elapsed Time[s]"], df["Data [Nm]"])
        ax.set_title("FM" + str(FM) + "\n" + test)
        ax.grid(True)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Torque [Nm]")
        ax.axvspan(start_time, end_time, alpha=0.4)
        
        # Display start and end of each phase
        for phase in df["Inner Phase"].unique():
            t_start_phase = df["Elapsed Time[s]"][df["Inner Phase"] == phase].iloc[0]
            ax.axvline(t_start_phase, alpha=0.5, color= "red", lw=2)
        
        # Display values next to graph
        ax.text(1.02, 1.00, description, va='top', transform=ax.transAxes)
        
        # Save the plot at given location
        if savepath is not None:
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            
            # Formatting the filename
            filename = "FM{}_{}.png".format(FM, test)
            filename = filename.replace("< ", "Pre-")
            filename = filename.replace("> ", "Post-")
            filename = filename.replace("(", "")
            filename = filename.replace(") [Nms/rad]", "")
            filename = filename.replace(") [Nm]", "")
            filename = filename.replace(" ", "_")
            
            fig.savefig(savepath + "/" + filename, bbox_inches='tight')
            plt.close(fig)
            

    def overview_raw_data(self, savepath=None):
        """Plot an overview of the rawdata."""
        fig, axs = plt.subplots(len(self.step_names), 1, figsize=(8, 4* len(self.step_names)))

        for i, test in enumerate(self.step_names):
            for FM in self.raw_files.index:
                if self.raw_data_dict[FM][test] is not None:
                    axs[i].plot(self.raw_data_dict[FM][test]["Elapsed Time[s]"], 
                                    self.raw_data_dict[FM][test]["Data [Nm]"], 
                                    label="FM" + str(FM))
                axs[i].set_title(test) 
                axs[i].grid(True)
                axs[i].set_ylabel("Torque [Nm]")
        
        axs[-1].set_xlabel("Elapsed Time [s]")
        
        # Save the plot
        if savepath is not None:
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            fig.savefig(savepath + "/overview_raw_data.png", bbox_inches='tight')
        
        
    def overview_processed_data(self, savepath=None):
        """Plot an overview of the processed data."""
        n = len(self.processed_data_dict)
        fig, axs = plt.subplots(n, 1, figsize=(7, n*3.5))
        for i, (FM, FM_dict) in enumerate(self.processed_data_dict.items()):
            for test in FM_dict.keys():
                if "ST" in test:
                    axs[i].plot(FM_dict[test], label=test)
            axs[i].legend(loc='upper left', bbox_to_anchor=(1.02, 1))
            axs[i].set_ylim(0, 2.48)
            axs[i].axhline(1.24, color="red")
            axs[i].text(0.03, 0.9, "FM {}".format(FM), horizontalalignment='left', verticalalignment='center', transform=axs[i].transAxes)
            axs[i].grid(True)
            axs[i].set_ylabel("Torque [Nm]")
            axs[i].set_xlabel("Sample [-]")
        
        if savepath is not None:
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            fig.savefig(savepath + "/overview_processed_data.png", bbox_inches='tight')
            
    def _get_file_list(self, root_dir, subdirs, ignore_list):
        """[obselete] List all the files available to load """
        parasite_strings = ["_DS4-FMs", "FM", " (Japan)", "_Dubai", "_Plato"]
        summary_file_dict = {}
        raw_data_dict = {}
        for subdir in subdirs:
            data_dir = root_dir + "/" + subdir
            for element in os.listdir(data_dir):
                element_path = data_dir + "/" + element
                if os.path.isdir(element_path) and element[0] != "0" and element not in ignore_list:
                    # Clean the directory name to only contain FM numbers
                    FM_tuple = element
                    for s in parasite_strings:
                        FM_tuple = FM_tuple.replace(s, "")
                    FM_tuple = tuple([int(fm) for fm in FM_tuple.replace("_", ",").split(",")])
                    raw_data_dict[FM_tuple] = []
                    
                    # Iterate over subelements conatining several FMs to get summary file paths
                    subelements = os.listdir(element_path)
                    for subelement in subelements:
                        subelement_path = element_path + "/" + subelement
                        if "Summary" in subelement or "Miscellaneous" in subelement:
                            for sheet in os.listdir(subelement_path):
                                if "~" not in sheet and "FM" in sheet and ".xlsx" in sheet and sheet not in ignore_list:
                                    FM = sheet.split("FM")[1][:3]
                                    if FM in summary_file_dict:
                                        print("Warning, multiple summary seet for FM {}".format(FM))
                                    summary_file_dict[int(FM)] = subelement_path + "/" + sheet
                    # Walk through directory to obtain all the raw data (csv) file paths
                    for path, directories, files in os.walk(element_path):
                        if path.split("\\")[-1]== "Data":
                            raw_data_dict[FM_tuple] += [path + "/" + f for f in files]
                                
        return summary_file_dict, raw_data_dict

    def _get_template(self, file):
        """[obselete] Infer the template of a given file"""
        observed_value = pd.read_excel(file, skiprows=1, usecols="C")
        for key, value in self.template_dict.items():
            if value.equals(observed_value):
                return key
        raise ValueError("Template not found.")
        
    def _create_template_dict(self):
        """[obselete] Create the list of templates."""
        print("Gathering all the templates...", end=" ")
        self.template_dict = {}
        template_dict_tmp = {}
        for FM, file in self.files.items():
            template_dict_tmp[FM] = pd.read_excel(file, skiprows=1, usecols="C")
        for key, value in template_dict_tmp.items():
            seen = False
            for key_seen, value_seen in self.template_dict.items():
                if value_seen.equals(value):
                    seen = True
            if not seen:
                self.template_dict[len(self.template_dict) + 1] = value
        
        print("Done")
