#!/ur/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import TP1_functions as lib

def produire_image(nom,f,g,lamb,mu,nitermax,stepini):
    ux,uy,CF,step=lib.RecalageDG_TP(f,g,lamb,mu,nitermax,stepini)
    plt.rcParams['text.usetex'] = 'true'
    plt.rcParams['text.latex.unicode'] = 'true'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 9
    fig, ax = plt.subplots(2,3)
    ax[0,0].imshow(f, origin='lower')
    ax[0,0].set_title('original function')
    ax[0,1].imshow(g, origin='lower')
    ax[0,1].set_title('target function')

    ax[1,0].quiver(uy,ux, color='b')
    ax[1,0].set_title('displacement field')
    ax[1,1].imshow(lib.interpol(f,ux,uy), origin='lower')
    ax[1,1].set_title('final function')
    ax[0,2].plot(CF)
    ax[0,2].set_title('objective history')
    ax[1,2].plot(np.log(step))
    ax[1,2].set_title('step history (log scale)')
    filename = nom+"-lam_"+str(lamb)+"-mu_"+str(mu)+"-stepini_"+str(stepini)+"-iter_"+str(step.size)+".png"    
    fig.savefig(filename, bbox_inches='tight')


#ux,uy,CF,step=lib.RecalageDG_TP(f,g,lamb,mu,nitermax,stepini)
#produire_image("simple_fonction",f,g,lamb,mu,stepini,nitermax)


#from PIL import Image
# ATTENTION, il faut installer Pillow: `pip install Pillow`
from scipy import misc as scipymisc 
def tester_avec_images_reelles():
    # Une dimension est une composante. Ici, RGB = 3 dimensions
    # On est en format ligne * colonnes * points, on veut passer
    # en une simple liste de points.
    f = np.array(scipymisc.imread('IRM1.png').tolist())
    f = f[:,:351]
    g = np.array(scipymisc.imread('IRM2.png').tolist())
    g = g[:,:351]
    produire_image("irm",f,g,lamb,mu,stepini,nitermax)

#from joblib import Parallel, delayed
def tester_image_simple():
    f,g=lib.get_images()
    lamb=[0,10,50,70,100]
    mu=[0,10,50,70,100]
    stepini=[0.001]
    nitermax=[100000]
    for m in mu:
        for s in stepini:
            for n in nitermax:
		for l in lamb:
                	produire_image("simple",f,g,l,m,n,s)
                #Parallel(n_jobs=4)(delayed(produire_image)("simple",f,g,l,m,n,s) for l in lamb)



if __name__ == '__main__':
    f,g=lib.get_images()
    produire_image("simple",f,g,50,50,1000,0.001)
    #tester_image_simple()

