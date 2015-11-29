import numpy as np
import matplotlib.pyplot as plt
import TP1_functions as lib


f,g=lib.get_images()


lamb=10
mu=10
nitermax=500
stepini=0.001
ux,uy,CF,step=lib.RecalageDG_TP(f,g,lamb,mu,nitermax,stepini)

fig, ax = plt.subplots(2,3)
ax[0,0].imshow(f)
ax[0,0].set_title('original function')
ax[0,1].imshow(g)
ax[0,1].set_title('target function')
ax[1,0].quiver(ux,uy)
ax[1,0].set_title('displacement field')
ax[1,1].imshow(lib.interpol(f,ux,uy))
ax[1,1].set_title('final function')
ax[0,2].plot(CF)
ax[0,2].set_title('objective history')
ax[1,2].plot(np.log(step))
ax[1,2].set_title('step history (log scale)')
plt.show()
while not plt.waitforbuttonpress():
    None



