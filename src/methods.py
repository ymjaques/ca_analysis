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

def calc_R(xc, yc,x,y):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def algmeth(xdp,ydp,xhi):
    """algebraic method"""
    x=xdp[ydp>xhi]
    y=ydp[ydp>xhi]

    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # linear system defining the center (uc, vc) in reduced coordinates:
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv  = np.sum(u*v)
    Suu  = np.sum(u**2)
    Svv  = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)

    # Solving the linear system
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = linalg.solve(A, B)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # Calcul des distances au centre (xc_1, yc_1)
    Ri_1     = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1      = np.mean(Ri_1)
    residu_1 = np.sum((Ri_1-R_1)**2)
    return xc_1,yc_1,R_1

def scipleastsq(xdp,ydp,xhi):
    """scipy least squares method"""
    x=xdp[ydp>xhi]
    y=ydp[ydp>xhi]

    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    #  == METHOD 2 ==
    from scipy      import optimize
#     from math import sqrt

#     method_2 = "leastsq"

#     def calc_R(xc, yc):
#         """ calculate the distance of each 2D points from the center (xc, yc) """
#         return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2       = calc_R(*center_2)
    R_2        = Ri_2.mean()
    residu_2   = sum((Ri_2 - R_2)**2)
    return xc_2,yc_2,R_2

def odrmeth(xdp,ydp,xhi):
    """ODR method"""
    x=xdp[ydp>xhi]
    y=ydp[ydp>xhi]

    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    # == METHOD 3 ==
    from scipy      import  odr

    method_3 = "odr"

    def f_3(beta, x):
        """ implicit definition of the circle """
        return (x[0]-beta[0])**2 + (x[1]-beta[1])**2 -beta[2]**2

    # initial guess for parameters
    R_m = calc_R(x_m, y_m,x,y).mean()
    beta0 = [ x_m, y_m, R_m]

    # for implicit function :
    #       data.x contains both coordinates of the points (data.x = [x, y])
    #       data.y is the dimensionality of the response
    lsc_data  = odr.Data(np.row_stack([x, y]), y=1)
    lsc_model = odr.Model(f_3, implicit=True)
    lsc_odr   = odr.ODR(lsc_data, lsc_model, beta0)
    lsc_out   = lsc_odr.run()

    xc_3, yc_3, R_3 = lsc_out.beta
    Ri_3 = calc_R(xc_3, yc_3,x,y)
    residu_3 = sum((Ri_3 - R_3)**2)
    return xc_3,yc_3,R_3 

def just_dens(datafile,trjfile,delta=0.5,start=1):
    """
    Module to just get the density grid, no contact angle calculation.
    """
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
    
    return drop_d.results.density,box