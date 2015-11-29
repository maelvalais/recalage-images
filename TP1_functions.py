#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt

def get_images() :
    n=21
    sigma=0.3
    [X,Y]=np.meshgrid(np.linspace(-1,1,n),np.linspace(-1,1,n))
    Z=np.sqrt(X*X+Y*Y)
    im1=np.zeros((n,n))
    im1[Z<=.7]=1.
    im1[Z<=.3]=.5
    im1[Z<=.1]=.7
    im2=np.zeros((n,n));
    Z=np.sqrt((X-.3)**2+(Y+.2)**2)
    im2[Z<=.7]=1
    im2[Z<=.3]=.5
    im2[Z<=.1]=.7
    G=np.fft.fftshift(np.exp(-(X**2+Y**2)/sigma**2))
    f=np.real(np.fft.ifft2(np.fft.fft2(G)*np.fft.fft2(im1)))
    g=np.real(np.fft.ifft2(np.fft.fft2(G)*np.fft.fft2(im2))) 
    f=f/np.max(f)
    g=g/np.max(g)
    return f,g

# Retourne: la matrice interpolée des décalages u(x)i,j = (u_x,u_y)i,j
# pour tout x(i,j) de l'image f.
# Sachant que u \in H^1 : R^2 -> R^2 donc pour chaque valeur de l'image
# on doit stocker un couple.
# Donc pour tout i,j de l'image, on prend x+u(x) et on interpole ça.
# Donc vu qu'on utilise le bilinéaire spline, ça fait pour tout i,j:
# - calcul de x+u
# - interpolation bilinéaire au point f(x+u) donc 4 éval de f + 3 barycentres
# Ce qui fait donc n * m * (4 + 3) calculs pour chaque appel.
def interpol(f,ux,uy) :
    # function that computes f \circ u and interpolates it on a mesh
    nx,ny=f.shape
    ip=interpolate.RectBivariateSpline(np.arange(nx),np.arange(ny),f)
    [X,Y]=np.meshgrid(np.arange(nx),np.arange(ny))
    X=X+ux
    Y=Y+uy
    return np.reshape(ip.ev(X.ravel(),Y.ravel()),(nx,ny))

# L'opérateur d donne toute la dérivée
#   d : (L^2)^2 -> R^2
#             u => d(u)
# Ici, on a divisé d en deux parties,
# du coup on a 
#   dx:     L^2 -> R
#            ux => dx(ux)
def dy(im) :
    d=np.zeros(im.shape)
    d[:,:-1] = im[:,1:] - im[:,:-1]
    return d

def dx(im) :
    d=np.zeros(im.shape)
    d[:-1,:] = im[1:,:] - im[:-1,:]
    return d
        
def dyT(im) :
    d=np.zeros(im.shape)
    d[:,1]    = -im[:,1]
    d[:,2:-1] = -dy(im)[:,1:-2]
    d[:,-1]   = im[:,-2]
    return d

def dxT(im) :
    d=np.zeros(im.shape)
    d[1,:]    = -im[1,:]
    d[2:-1,:] = -dx(im)[1:-2,:]
    d[-1,:]   = im[-2,:]
    return d

# Calcule E(u) + R(u) pour un u donné. Comme u:R^2->R^2,
# on doit donner (ux,uy) pour tout i,j de l'image.
# Note sur dx dy:
# Note: 
#   d : (H^1)^2 -> R^2
#             u => d(u)
#
#   dx:     H^1 -> R
#            ux => dx(ux)
#   
#   ux:     R^2 -> R
#             x => ux(x)
def objective_function(f,g,ux,uy,lamb,mu) :
    f_u = interpol(f,ux,uy) # f_u appartient à R^2 -> R
    #print f_u[0:10]
    return 1/2. * \
        (np.power(np.linalg.norm(f_u - g),2) \
        + mu * np.power(np.linalg.norm(dx(uy) + dy(ux)),2) \
        + (lamb+mu) * np.power(np.linalg.norm(dx(ux) + dy(uy)),2)), \
        f_u


def linesearch(ux,uy,step,descentx,descenty,obj_old,f,g,lamb,mu) :
    step=2*step
    tmpx=ux-step*descentx
    tmpy=uy-step*descenty
    obj,fu=objective_function(f,g,tmpx,tmpy,lamb,mu)
    while obj >obj_old and step > 1.e-8:
        step=0.5*step
        tmpx=ux-step*descentx
        tmpy=uy-step*descenty
        obj,fu=objective_function(f,g,tmpx,tmpy,lamb,mu)
    return tmpx,tmpy,step

def JTPsi(phi1,ph2,phi3,dfx,dfy,lamb,mu) :
    raise ValueError('to be implemented')  
        
def JTJ(p1,p2,dfx,dfy,lamb,mu,epsilon) :
    raise ValueError('to be implemented')  
    
def CGSolve(u0x,u0y,lamb,mu,b,epsilon,dfx,dfy) :
    # Solves JTJ[ux,uy]=b
    #lambd,mu,epsilon,dfx,dfy are needed in the computation of JTJ
    # [u0x,u0y] is the starting point of the iteration algo
    nitmax=100;
    ux=u0x;
    uy=u0y;
    # Computes JTJu
    Ax,Ay=JTJ(ux,uy,dfx,dfy,lamb,mu,epsilon);
    rx=b[0]-Ax
    ry=b[1]-Ay
    px=rx
    py=ry
    rsold=np.linalg.norm(rx)**2+np.linalg.norm(ry)**2
    for i in range(nitmax) :
        Apx,Apy=JTJ(px,py,dfx,dfy,lamb,mu,epsilon);
        alpha=rsold/(np.vdot(rx[:],Apx[:])+np.vdot(ry[:],Apy[:]))
        ux=ux+alpha*px
        uy=uy+alpha*py
        rx=rx-alpha*Apx
        ry=ry-alpha*Apy
        rsnew=np.linalg.norm(rx)**2+np.linalg.norm(ry)**2
        if np.sqrt(rsnew)<1e-10 :
            return [ux,uy]
        px=rx+rsnew/rsold*px
        py=ry+rsnew/rsold*py
        rsold=rsnew
    return [ux,uy]
        
def RecalageDG_TP(f,g,lamb,mu,nitermax,stepini) : 
    ux=np.zeros(f.shape)
    uy=np.zeros(f.shape)  
    CF=[]
    step_list=[]
    niter=0
    step=stepini
    while niter < nitermax and step > 1.e-8 : 
        niter+=1
        obj,fu=objective_function(f,g,ux,uy,lamb,mu)
        CF.append(obj)   
        # Gradient of F at point u = grad E + grad R
        # grad E = (f o (I+u) - g) * grad_f o (I+u)
        # grad R = (A + A^T)u
        # grad F \in (L^2(Omega))^2 -> (grad_F_x, grad_F_y)
        # grad_F_x = (f o (I+u) - g) * grad_fo(I+u)_x
        # grad_F_y = (f o (I+u) - g) * grad_fo(I+u)_y
        
        # NOTE: f_u correspond à "f o (I+u)"
        #       grad_f_u correspond à (drondx_f,drondy_f)
        
        # NOTE: je n'ai pas "optimisé" en créant des variables
        # intermédiaires (mutualiser les résultats) pour la lisibilité

        f_u = interpol(f,ux,uy) # f_u appartient à R^2 -> R
        grad_E_x = (f_u - g) * dx(f_u)
        grad_E_y = (f_u - g) * dy(f_u)
        
        grad_R_x = 1/2.*(lamb+mu)*(2*dxT(dx(ux)) + dyT(dy(uy))) + 1/2.*mu*(dyT(dy(ux)) + 2*dyT(dx(uy)))
        grad_R_y = 1/2.*(lamb+mu)*(2*dxT(dx(ux)) + dyT(dy(uy))) + 1/2.*mu*(2*dxT(dy(ux)) + 2*dxT(dx(uy)))

        gradx = grad_E_x + grad_R_x
        grady = grad_E_y + grad_R_y

        ux,uy,step=linesearch(ux,uy,step,gradx,grady,obj,f,g,lamb,mu)
        step_list.append(step)
        if (niter % 3 ==0) :
            print 'iteration :',niter,' cost function :',obj,'step :',step
    return ux,uy,np.array(CF),np.array(step_list)
          
def RecalageGN_TP(f,g,lamb,mu,nitermax,stepini,epsi) : 
    ux=np.zeros(f.shape)
    uy=np.zeros(f.shape)  
    descentx=np.zeros(f.shape)
    descenty=np.zeros(f.shape)  
    CF=[]
    step_list=[]
    niter=0
    step=stepini
    while niter < nitermax and step > 1.e-8 : 
        niter+=1
        obj,fu=objective_function(f,g,ux,uy,lamb,mu)
        CF.append(obj)
        # Gradient of F at point u
        raise ValueError('Compute b here')
        raise ValueError('Compute dfx,dfy here')    
        [descentx,descenty]=CGSolve(descentx,descenty,lamb,mu,b,epsi,dfx,dfy)
        ux,uy,step=linesearch(ux,uy,step,descentx,descenty,obj,f,g,lamb,mu)
        step_list.append(step)
        # Display
        if (niter % 3 ==0) :
            print 'iteration :',niter,' cost function :',obj,'step :',step
    return ux,uy,np.array(CF),np.array(step_list)
  
