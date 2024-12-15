import MDAnalysis
from MDAnalysis import Universe
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
import MDAnalysis.analysis.lineardensity as ld
from MDAnalysis.lib.log import ProgressBar
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import MDAnalysis.transformations
from MDAnalysis import Writer
from MDAnalysis.analysis.density import DensityAnalysis
from scipy import linalg
import pandas as pd
# import nglview as nv
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_ca(datafile,trjfile,delta=0.5,start=1):
    """calculate left and right contact angle of cilindrical droplet"""
    # define universe, drop selection, center trajectory, grid to use and run density analysis
    u=Universe(datafile,trjfile, format="LAMMPSDUMP",dt=0.001)
    drop=u.select_atoms('type 12 13')
    notdrop=u.select_atoms('not type 12 13')
    allatoms=u.select_atoms('all')
    box=u.dimensions[:3]
    workflow=[MDAnalysis.transformations.unwrap(drop),MDAnalysis.transformations.center_in_box(drop,center='geometry',wrap=False)]
    u.trajectory.add_transformations(*workflow)
    
    drop_d=DensityAnalysis(drop,delta=delta,gridcenter=box/2,xdim=box[0],ydim=box[1],zdim=box[2])
    drop_d.run(start=start,verbose=True)
    # center of drop to get the bulk density value
    xc,zc=int(np.round(box[0]/2)/delta),int(np.round(box[2]/2)/delta)
    # get a small sample from the middle of the drop, 10 Angstrom
    w=int(10/delta)
    # dm is the bulk density divided by two, our criterion to locate the l-v interface
    dm=drop_d.results.density.grid[xc-w:xc+w,:,zc-w:zc+w].mean()/2
#     dm=0.04
    # xlo is lowest z coordinate of the droplet, our defintion of the surface location
    # xlo=np.argmax(drop_d.results.density.grid[:,:,:].mean(axis=(0,1)).T)
#     xlo=np.nonzero(drop_d.results.density.grid[:,:,:].mean(axis=(0,1)).T)[0][0]
    xlo=np.where(drop_d.results.density.grid[:,:,:].mean(axis=(0,1))>1e-3)[0][0]
    # 10 Angstrom of the xlo is the place where we gonna consider for the fitting
    xhi=xlo+10/delta
    # copy of 3d density:
    dca=drop_d.results.density.grid[:,:,:].mean(axis=(1))
    # remove values greater than 10% of dm
    dcacuthi=np.where(dca<dm*1.1,dca,0)
    # remove values smaller than 10% of dm
    dcacutlo=np.where(dca>dm*0.9,dcacuthi,0)
    # just assigning as another name
    dcaf=dcacutlo
    # makes all values !=0 equal to 1
    dcaf1=np.where(dcaf!=0,dcaf,1)
    # put values in x and y variables
    xdp,ydp=np.where(dcaf1!=1)
    # fitting with algebraic method
    xc_1,yc_1,R_1=algmeth(xdp,ydp,xhi)
    # putting the results of fitting in a nice way
    theta_fit = np.linspace(-np.pi, np.pi, 180)
    x_fit1 = xc_1 + R_1*np.cos(theta_fit)
    y_fit1 = yc_1 + R_1*np.sin(theta_fit)
    # just assigning another variable for xlo
    hsurf=xlo
    # getting the fitted values only above hsurf
    xl,yl=xr,yr=x_fit1[y_fit1>hsurf],y_fit1[y_fit1>hsurf]
    # sorting the left side of the drop correctly
    xl,yl=xl[xl<xc],yl[xl<xc]
    xl,yl=xl[np.argsort(yl)],yl[np.argsort(yl)]
    # sorting the right side of the drop correctly
    xr,yr=xr[xr>xc],yr[xr>xc]
    xr,yr=xr[np.argsort(yr)],yr[np.argsort(yr)]
    # finally the ca values
    cal=np.degrees(np.arctan((yl[1]-yl[0])/(xl[1]-xl[0])))
    car=np.degrees(np.arctan((yr[1]-yr[0])/(xr[1]-xr[0])))
    if cal<0 and car>0:
        cal+=180
        car=180-car
    elif cal>0 and car<0:
        cal=cal
        car=-car
    else:
        print("Something is wrong with the ca's")
    return cal,car,x_fit1,y_fit1,drop_d.results.density.grid[:,:,:],box,hsurf