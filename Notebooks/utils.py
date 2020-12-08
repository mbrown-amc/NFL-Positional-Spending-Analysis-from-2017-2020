import os
import pandas as pd

def get_data(dtype):
    """
    Loads the data for the project.
    :param dtype: String. The type of data desired.
    
    """
    
    import os
    import pandas as pd
    
    pardir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    datadir = pardir + "\\Data"
    datadir
    
    if dtype == "stats":
    
        offense = []
        defense = []

        offensel = []
        defensel = []

        for f in os.listdir(datadir):
            if f[11] == "O":
                offense.append(datadir + "\\" + f)
            if f[11] == "D":
                defense.append(datadir + "\\" + f)

        for f in offense:
            offensel.append(pd.read_excel(f,0))
        for f in defense:
            defensel.append(pd.read_excel(f,0))

        return offensel, defensel
    
    elif dtype == "salary":
        
        data = datadir + "\\Salary_Data_Players.xlsx"
     
        salary = pd.read_excel(data,0)
        
        salary["% OF YEARLY CAP"] = salary["% OF YEARLY CAP"] * 100
        
        return salary
    
    elif dtype == "wins":
        
        AFC = []
        NFC = []

        AFCl = []
        NFCl = []
    
        for f in os.listdir(datadir):
            if f[5] == "A":
                AFC.append(datadir + "\\" + f)
            if f[5] == "N":
                NFC.append(datadir + "\\" + f)

        for f in AFC:
            AFCl.append(pd.read_excel(f,0))
        for f in NFC:
            NFCl.append(pd.read_excel(f,0))
        return AFCl, NFCl
        
def clean_data(dtype, offense = False, defense = False, AFCl = False, NFCl = False, test = False):
    """
    Prepares the data for the project.
    :param dtype: String. The type of data being passed.
    :param offense: DataFrame. The offense team data.
    :param defense: DataFrame. The defense team data.
    :param AFCl: DataFrame. The AFC team data for win percentage.
    :param NFCl: DataFrame. The NFC team data for win percentage.
    :param test: DataFrame. The positional salary data.
    """
    import pandas as pd
    
    if dtype == "stats":
    
        i = 2017

        for f in offense:
            f.insert(0, "Year", i)
            i+= 1

        j = 2017

        for g in defense:
            g.insert(0, "Year", j)
            j+= 1
        for i in range(0, len(offense)):
            offense[i] = offense[i].drop([32,33,34])

        for i in range(0, len(defense)):
            defense[i] = defense[i].drop([32,33,34])

        combined = []
        i = 0
        for f in offense:
            combined.append(f.merge(defense[i], how='inner', on=["Year","Tm"]))
            i+=1
            
        finalframe = pd.concat(combined, ignore_index=True)
        data = finalframe.drop(["G_x", "G_y", "Rk_x", "Rk_y"], axis = 1)
        data.loc[(data.Tm == "Arizona Cardinals"), "Tm"] = 'ARI'
        data.loc[(data.Tm == "Atlanta Falcons"), "Tm"] = 'ATL'
        data.loc[(data.Tm == "Baltimore Ravens"), "Tm"] = 'BAL'
        data.loc[(data.Tm == "Buffalo Bills"), "Tm"] = 'BUF'
        data.loc[(data.Tm == "Carolina Panthers"), "Tm"] = 'CAR'
        data.loc[(data.Tm == "Chicago Bears"), "Tm"] = 'CHI'
        data.loc[(data.Tm == "Cincinnati Bengals"), "Tm"] = 'CIN'
        data.loc[(data.Tm == "Cleveland Browns"), "Tm"] = 'CLE'
        data.loc[(data.Tm == "Dallas Cowboys"), "Tm"] = 'DAL'
        data.loc[(data.Tm == "Denver Broncos"), "Tm"] = 'DEN'
        data.loc[(data.Tm == "Detroit Lions"), "Tm"] = 'DET'
        data.loc[(data.Tm == "Green Bay Packers"), "Tm"] = 'GB'
        data.loc[(data.Tm == "Houston Texans"), "Tm"] = 'HOU'
        data.loc[(data.Tm == "Indianapolis Colts"), "Tm"] = 'IND'
        data.loc[(data.Tm == "Jacksonville Jaguars"), "Tm"] = 'JAX'
        data.loc[(data.Tm == "Kansas City Chiefs"), "Tm"] = 'KC'
        data.loc[(data.Tm == "Los Angeles Chargers"), "Tm"] = 'LAC'
        data.loc[(data.Tm == "Los Angeles Rams"), "Tm"] = 'LAR'
        data.loc[(data.Tm == "Miami Dolphins"), "Tm"] = 'MIA'
        data.loc[(data.Tm == "Minnesota Vikings"), "Tm"] = 'MIN'
        data.loc[(data.Tm == "New England Patriots"), "Tm"] = 'NE'
        data.loc[(data.Tm == "New Orleans Saints"), "Tm"] = 'NO'
        data.loc[(data.Tm == "New York Giants"), "Tm"] = 'NYG'
        data.loc[(data.Tm == "New York Jets"), "Tm"] = 'NYJ'
        data.loc[(data.Tm == "Oakland Raiders"), "Tm"] = 'LV'
        data.loc[(data.Tm == "Las Vegas Raiders"), "Tm"] = 'LV'
        data.loc[(data.Tm == "Philadelphia Eagles"), "Tm"] = 'PHI'
        data.loc[(data.Tm == "Pittsburgh Steelers"), "Tm"] = 'PIT'
        data.loc[(data.Tm == "San Diego Chargers"), "Tm"] = 'SD'
        data.loc[(data.Tm == "San Francisco 49ers"), "Tm"] = 'SF'
        data.loc[(data.Tm == "St. Louis Rams"), "Tm"] = 'STL'
        data.loc[(data.Tm == "Seattle Seahawks"), "Tm"] = 'SEA'
        data.loc[(data.Tm == "Tampa Bay Buccaneers"), "Tm"] = 'TB'
        data.loc[(data.Tm == "Tennessee Titans"), "Tm"] = 'TEN'
        data.loc[(data.Tm == "Washington Redskins"), "Tm"] = 'WAS'
        data.loc[(data.Tm == "Washington Football Team"), "Tm"] = 'WAS'
        
        data = data.rename(columns = {"Year": "YEAR", "Tm" : "TEAM"})
        return data
    
    elif dtype == "wins":
        
        i = 2017

        for j in AFCl:
            j.insert(0, "Year", i)
            i+= 1

        i = 2017

        for k in NFCl:
            k.insert(0, "Year", i)
            i+= 1
            
        for t in AFCl:
            t.Tm = t.Tm.str.strip("*")
            t.Tm = t.Tm.str.strip("+")
            
        for t in NFCl:
            t.Tm = t.Tm.str.strip("*")
            t.Tm = t.Tm.str.strip("+")
            
        AFCf = pd.concat(AFCl, ignore_index=True)
        NFCf = pd.concat(NFCl, ignore_index=True)
        
        ff = (AFCf, NFCf)
        data = pd.concat(ff)
        data.loc[(data.Tm == "Arizona Cardinals"), "Tm"] = 'ARI'
        data.loc[(data.Tm == "Atlanta Falcons"), "Tm"] = 'ATL'
        data.loc[(data.Tm == "Baltimore Ravens"), "Tm"] = 'BAL'
        data.loc[(data.Tm == "Buffalo Bills"), "Tm"] = 'BUF'
        data.loc[(data.Tm == "Carolina Panthers"), "Tm"] = 'CAR'
        data.loc[(data.Tm == "Chicago Bears"), "Tm"] = 'CHI'
        data.loc[(data.Tm == "Cincinnati Bengals"), "Tm"] = 'CIN'
        data.loc[(data.Tm == "Cleveland Browns"), "Tm"] = 'CLE'
        data.loc[(data.Tm == "Dallas Cowboys"), "Tm"] = 'DAL'
        data.loc[(data.Tm == "Denver Broncos"), "Tm"] = 'DEN'
        data.loc[(data.Tm == "Detroit Lions"), "Tm"] = 'DET'
        data.loc[(data.Tm == "Green Bay Packers"), "Tm"] = 'GB'
        data.loc[(data.Tm == "Houston Texans"), "Tm"] = 'HOU'
        data.loc[(data.Tm == "Indianapolis Colts"), "Tm"] = 'IND'
        data.loc[(data.Tm == "Jacksonville Jaguars"), "Tm"] = 'JAX'
        data.loc[(data.Tm == "Kansas City Chiefs"), "Tm"] = 'KC'
        data.loc[(data.Tm == "Los Angeles Chargers"), "Tm"] = 'LAC'
        data.loc[(data.Tm == "Los Angeles Rams"), "Tm"] = 'LAR'
        data.loc[(data.Tm == "Miami Dolphins"), "Tm"] = 'MIA'
        data.loc[(data.Tm == "Minnesota Vikings"), "Tm"] = 'MIN'
        data.loc[(data.Tm == "New England Patriots"), "Tm"] = 'NE'
        data.loc[(data.Tm == "New Orleans Saints"), "Tm"] = 'NO'
        data.loc[(data.Tm == "New York Giants"), "Tm"] = 'NYG'
        data.loc[(data.Tm == "New York Jets"), "Tm"] = 'NYJ'
        data.loc[(data.Tm == "Oakland Raiders"), "Tm"] = 'LV'
        data.loc[(data.Tm == "Las Vegas Raiders"), "Tm"] = 'LV'
        data.loc[(data.Tm == "Philadelphia Eagles"), "Tm"] = 'PHI'
        data.loc[(data.Tm == "Pittsburgh Steelers"), "Tm"] = 'PIT'
        data.loc[(data.Tm == "San Diego Chargers"), "Tm"] = 'SD'
        data.loc[(data.Tm == "San Francisco 49ers"), "Tm"] = 'SF'
        data.loc[(data.Tm == "St. Louis Rams"), "Tm"] = 'STL'
        data.loc[(data.Tm == "Seattle Seahawks"), "Tm"] = 'SEA'
        data.loc[(data.Tm == "Tampa Bay Buccaneers"), "Tm"] = 'TB'
        data.loc[(data.Tm == "Tennessee Titans"), "Tm"] = 'TEN'
        data.loc[(data.Tm == "Washington Redskins"), "Tm"] = 'WAS'
        data.loc[(data.Tm == "Washington Football Team"), "Tm"] = 'WAS'
        
        data2 = data[["Year","Tm", "W-L%"]]
        data2 = data2.rename(columns = {"Year": "YEAR", "Tm" : "TEAM", "W-L%": "W%"})
        
        return data2
    elif dtype == "salary":
        testgroup = test.groupby(["TEAM", "YEAR", "POS"]).sum()
        pos = test.POS.unique()
        team = test.TEAM.unique()
        year = test.YEAR.unique()
        tlist = []
        for t in team:
            for y in year:
                for p in pos:
                    try:
                        tlist.append(testgroup.loc[(t,y,p), "% OF YEARLY CAP"])
                    except:
                        tlist.append(0)               
        mteam = []
        for t in team:
            for y in year:
                mteam.append(t)
                
        myear = []
        for t in team:
            for y in year:
                myear.append(y)
       
        data = {"YEAR": myear, "TEAM": mteam}
        newdf = pd.DataFrame(data)
        for p in pos:
            newdf[p] = 0
            
        c = 0
        r = 0
        n = 0

        for i in range(len(tlist)):
            newdf.iloc[r, c+2] = tlist[n]
            c += 1
            n+=1
            if c > len(pos)-1:
                c = 0
                r += 1
                
        return newdf

import matplotlib.pyplot as plt
import numpy as np
    
def find_clusters(ClusterTeams):
    """
    Finds the optimal number of clusters using KMeans.
    :param ClusterTeams: DataFrame. The data to find the number of clusters of.
    """
    
    from yellowbrick.cluster import KElbowVisualizer
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(random_state = 52594)

    visualizer = KElbowVisualizer(kmeans, k=(2,10), metric = 'calinski_harabasz', timings = False)

    visualizer.fit(ClusterTeams)
    visualizer.show()
    
    kmeans = KMeans(random_state = 52594)

    visualizer = KElbowVisualizer(kmeans, k=(2,10), metric = 'silhouette', timings = False)

    visualizer.fit(ClusterTeams)
    visualizer.show()
    
def scale_data(data):
    """
    Scales the data using StandardScaler
    :param data: DataFrame like object. The data to be scaled.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns = data.columns)
    return scaled_data
    
def cluster_data(data, numclusters = 3):
    """
    Clusters the data using KMeans.Returns the cluster predictions on the data.
    :param data: DataFrame like object. The data to perform clustering on.
    :param numclusters: Int. The number of clusters to make for clustering.

    """
    
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters = numclusters, random_state = 52594)

    kmeans.fit(data)

    clusters = kmeans.predict(data)
    
    return clusters
    
    
def add_clusters(data, clusters):
    """
    Adds the cluster predictions to the original data for interpretation.
    :param data: DataFrame. The data to have the cluster predictions added on to.
    :param clusters: List. The list of cluster predictions to be added to the DataFrame.
    """
    addclusters = data
    addclusters["cluster"] = clusters
    return addclusters

def pca_exp_var(data):
    """
    Charts the explained variance per component from PCA.
    :param data: DataFrame. The data for PCA to show the explained variance per component of.
    """
    from sklearn.decomposition import PCA
    pca = PCA(random_state = 52594).fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('num comp')
    plt.ylabel('cumul expl var');
    
def pca(data, numcomp = .99):
    """
    Performs PCA on the given data.
    :param data: DataFrame. The data to perform PCA on.
    :Param numcomp: Variable. As an int, the number of components to use when performing PCA. As a 2 decimal float < 1, the percentage of explained variance required when PCA is complete.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=numcomp, random_state = 52594)

    pca.fit(data)

    print("the explained variance ratio is ", pca.explained_variance_ratio_.sum())
    
    reduct = pca.transform(data)

    print("The shape of the original data is ", data.shape)
    print("The shape after pca is ", reduct.shape)
    return reduct

def plot(cluster0, cluster1, cluster2, cluster3, cluster4, cluster5, x, y, xlabel, ylabel, l1, l2, l3, l4, l5, l6):
    """
    Plots data.
    :param cluster0: DataFrame. The data belonging to the first cluster.
    :param cluster1: DataFrame. The data belonging to the second cluster.
    :param cluster2: DataFrame. The data belonging to the third cluster.
    :param cluster3: DataFrame. The data belonging to the forth cluster.
    :param cluster4: DataFrame. The data belonging to the fifth cluster.
    :param cluster5: DataFrame. The data belonging to the sixth cluster.
    :param x: variable. The x column to be plotted.
    :param y: variable. The y column to be plotted.
    :param xlabel: String. The x label for the plot.
    :param ylabel: String. The y label for the plot.
    :param l1: String. The label for the first cluster.
    :param l2: String. The label for the second cluster.
    :param l3: String. The label for the third cluster.
    :param l4: String. The label for the forth cluster.
    :param l5: String. The label for the fifth cluster.
    :param l6: String. The label for the sixth cluster.
    """
    figure = plt.figure()
    plot = figure.add_subplot(111)
    
    plt.scatter(cluster0[x], cluster0[ y], s=10, c='r', cmap = "rainbow", marker = "s", label = l1)
    plt.scatter(cluster1[x], cluster1[ y], s=10, c='b', cmap = "rainbow", marker = "s", label = l2)
    plt.scatter(cluster2[x], cluster2[ y], s=10, c='g', cmap = "rainbow", marker = "s", label = l3)
    plt.scatter(cluster3[x], cluster3[ y], s=10, c='y', cmap = "rainbow", marker = "s", label = l4)
    plt.scatter(cluster4[x], cluster4[ y], s=10, c='orange', cmap = "rainbow", marker = "s", label = l5)
    plt.scatter(cluster5[x], cluster5[ y], s=10, c='black', cmap = "rainbow", marker = "s", label = l6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    
def break_clusters(data):
    """
    Breaks the data into clusters for use in graphing.
    :param data: DataFrame. The data to be broken into clusters.
    """
    import pandas as pd

    c0 = data.loc[data["cluster"] == 0]
    c1 = data.loc[data["cluster"] == 1]
    c2 = data.loc[data["cluster"] == 2]
    c3 = data.loc[data["cluster"] == 3]
    c4 = data.loc[data["cluster"] == 4]
    c5 = data.loc[data["cluster"] == 5] 
    return c0, c1, c2, c3, c4, c5