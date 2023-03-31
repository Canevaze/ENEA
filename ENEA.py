from math import *
from sklearn.cluster import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from scipy.linalg import lstsq


file = pd.read_parquet('Downloads/lidar_cable_points_easy.parquet',engine='fastparquet')

# Exact the columns as np array

x = file['x'].values
y = file['y'].values
z = file['z'].values

# Plot function for 3D points

def plot3d(x,y,z, label=None):

    if label is None:
        label = [0] * len(x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=label, marker='o', cmap='rainbow')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

# Plot the 3D points

#plot3d(x,y,z)

# Clustering function With K means (we currently do not use it)

def K_clustering(x,y,z, n_clusters=3):

    clf = KMeans(n_clusters=n_clusters)
    X = x
    Y = y
    Z = z
    d= {'X' : X, 'Y' : Y, 'Z' : Z}
    df = pd.DataFrame(data = d)
    clf.fit(df)
    labels = clf.labels_
    return labels

# Clustering with DBscan 

def DB_clustering(x,y,z, eps=0.1, min_samples=10):

    clf = DBSCAN(eps=eps, min_samples=min_samples)
    X = x
    Y = y
    Z = z
    d= {'X' : X, 'Y' : Y, 'Z' : Z}
    df = pd.DataFrame(data = d)
    clf.fit(df)
    labels = clf.labels_
    return labels

# Divide the points between the clusters 

def divide_points(x,y,z, labels, n):
    clusters = []
    for i in range(n):
        clusters.append([])
    
    for i in range(len(x)):
        clusters[labels[i]].append([x[i],y[i],z[i]])
    
    return clusters
    

# Plot the 3D points with the clustering

labels_K = K_clustering(x,y,z, n_clusters=3)
labels_DB = DB_clustering(x,y,z, eps=0.65, min_samples=2)

# Get the clusters from labels_DB and print them 

clusters = divide_points(x,y,z, labels_DB, 3)

# get average vector on plan X,Y for each cluster 

def average_vector(clusters):
    X = []
    for i in range(len(clusters)):
        
        # Use regression to get the average vector on the plan X,Y

        # Get the points of the cluster
        points = clusters[i]

        # Get the points coordinates
        x = [points[j][0] for j in range(len(points))]
        y = [points[j][1] for j in range(len(points))]

        # Average axe min square error in 2D
        A = np.array([x,np.ones(len(x))]).T
        a,b = lstsq(A,y)[0]

        # return the average vector
        X.append([a,b])
    
    return X

# Plot 2D average vector on plan X,Y for each cluster

def plot_average_vector(clusters):
    
    X = average_vector(clusters)

    fig = plt.figure()

    # Using subplots compare each cluster with the average vector
    for i in range(len(clusters)):
        ax = fig.add_subplot(1, len(clusters), i+1)
        ax.scatter([clusters[i][j][0] for j in range(len(clusters[i]))], [clusters[i][j][1] for j in range(len(clusters[i]))])
        
        x = np.linspace(-10,10,100)
        y = X[i][0]*x + X[i][1]
        ax.plot(x,y)

        ax.set_title("Cluster "+str(i))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    
    plt.show()

# optimize x0, z0 and c to get the best fit using min square error

def optimize(clusters):

    X = average_vector(clusters)
    a_list = [X[i][0] for i in range(len(X))]
    b_list = [X[i][1] for i in range(len(X))]

    X0 = []
    Z0 = []
    C = []

    # Get the points of the cluster
    for i in range(len(clusters)):

        # Get the points coordinates 
        x = [clusters[i][j][0] for j in range(len(clusters[i]))]
        y = [clusters[i][j][1] for j in range(len(clusters[i]))]
        z = [clusters[i][j][2] for j in range(len(clusters[i]))]

        x  = np.array(x)
        y  = np.array(y)
        z  = np.array(z)

        # Define the function to minimize
        def f(x0, x,y,z, a0,b0):
            p1 = x0[0]
            p2 = x0[1]
            p3 = x0[2]
            y_ = a0*x + b0
            z_ = p2 + p3*(np.cosh((x-p1)/p3)-1)
            return np.sum((y-y_)**2 + (z-z_)**2)
        
        # Get the best fit
        param = minimize(f, x0=[0,0,0.1], args=(x,y,z, a_list[i], b_list[i]), method='Nelder-Mead', tol=1e-6)
            
        # Get the best fit parameters
        x0 = param.x[0]
        z0 = param.x[1]
        c = param.x[2]

        # Add the best fit parameters to the list
        X0.append(x0)
        Z0.append(z0)
        C.append(c)

    return X0, Z0, C

def f(x,y,z, a0,b0, param):
    x0 = param[0]
    z0 = param[1]
    c = param[2]
    y_ = a0*x + b0
    z_ = z0 + c*(np.cosh((x-x0)/c)-1)
    return np.sum((y-y_)**2 + (z-z_)**2)

# Print the estimated function 

def plot_function(clusters, X, X0, Z0, C):
    
    # Plot the points on the graph 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the points of the cluster
    for i in range(len(clusters)):
        # Get the points coordinates 
        x = [clusters[i][j][0] for j in range(len(clusters[i]))]
        y = [clusters[i][j][1] for j in range(len(clusters[i]))]
        z = [clusters[i][j][2] for j in range(len(clusters[i]))]

        x  = np.array(x)
        y  = np.array(y)
        z  = np.array(z)

        # Get the best fit parameters
        x0 = X0[i]
        z0 = Z0[i]
        c = C[i]

        # Get the best fit function
        x_min = np.min(x)
        x_max = np.max(x)
        x_ = np.linspace(x_min,x_max,1000)
        y_ = X[i][0]*x_ + X[i][1]
        z_ = z0 + c*(np.cosh((x_-x0)/c)-1)

        # Plot the points and the function
        ax.scatter(x, y, z, marker='o')
        ax.plot(x_, y_, z_)

    plt.show()

X = average_vector(clusters)

# visualize the average vector on plan X,Y for each cluster
#plot_average_vector(clusters)

X0, Z0, C = optimize(clusters)
plot_function(clusters, X, X0, Z0, C)

#plot3d(x,y,z, label=labels_DB)
