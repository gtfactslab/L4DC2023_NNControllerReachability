import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse
from matplotlib.collections import LineCollection

def file_to_numpy (filenames) :
    with open('data/' + filenames[0] + '.npy', 'rb') as f :
        nploaded = np.load(f)
        X = nploaded['X']
        U = nploaded['U']
    for FILE in filenames[1:] :
        with open('data/' + FILE + '.npy', 'rb') as f :
            nploaded = np.load(f)
            X = np.append(X, nploaded['X'], axis=0)
            U = np.append(U, nploaded['U'], axis=0)
    return X,U

def numpy_to_file (X, U, filename) :
    with open(filename, 'wb') as f :
        np.savez(f, X=X, U=U)

def uniform_disjoint (set, N) :
    probs = [s[1] - s[0] for s in set]; probs = probs / np.sum(probs)
    return np.array([np.random.choice([
        np.random.uniform(s[0], s[1]) for s in set
    ], p=probs) for _ in range(N)])

def gen_ics(XRANGE, YRANGE, PRANGE, VRANGE, N) :
    X = np.empty((N, 4))
    X[:,0] = uniform_disjoint(XRANGE, N)
    X[:,1] = uniform_disjoint(YRANGE, N)
    X[:,2] = uniform_disjoint(PRANGE, N)
    X[:,3] = uniform_disjoint(VRANGE, N)
    return X


def plot_XY_t (ax, tt, XX, YY, dXX=None, dYY=None, dXXh=None, dYYh=None) :
    ax.plot(tt, XX, label="X", color='C0')
    ax.plot(tt, YY, label="Y", color='C1')
    if dXX is not None and dYY is not None and dXXh is not None and dYYh is not None :
        ax.plot(tt, dXX,  color='C0', label='dx', linewidth=0.5)
        ax.plot(tt, dXXh, color='C0', label='dxh',linewidth=0.5)
        ax.fill_between(tt, dXX, dXXh, color='C0', alpha=1)
        ax.plot(tt, dYY,  color='C1', label='dy', linewidth=0.5)
        ax.plot(tt, dYYh, color='C1', label='dyh',linewidth=0.5)
        ax.fill_between(tt, dYY, dYYh, color='C1', alpha=1)
    ax.legend()
    ax.set_title("x,y vs t")
def plot_PV_t (ax, tt, PP, VV) :
    ax.plot(tt, PP, label="psi", color='C2')
    ax.plot(tt, VV, label="v", color='C3')
    ax.legend()
    ax.set_title("psi,v vs t")
def plot_XYPV_t (ax, tt, SS) :
    XX, YY, PP, VV = SS
    ax.plot(tt, XX, label="X")#, color='C0')
    ax.plot(tt, YY, label="Y")#, color='C1')
    ax2 = ax.twinx()
    ax2.plot(tt, PP, label="psi", color='C2')
    ax2.plot(tt, VV, label="v", color='C3')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2)
    ax.set_xlabel('t',labelpad=0)
    ax.set_ylabel('x,y',labelpad=0)
    ax2.set_ylabel('psi,v',labelpad=0)
    ax.set_title("state vs t")
    # ax.set_xlim([1,1.5])
    # ax.set_ylim([-15,15])
    # ax.set_ylim([6,10])
def plot_Y_X (fig, ax, tt, XX, YY, xlim=[-15,15], ylim=[-15,15], show_colorbar=False, show_obs=True) :
    if show_obs :
        ax.add_patch(Circle((4,4),3/1.25,lw=0,fc='darkred'))
        ax.add_patch(Circle((-4,4),3/1.25,lw=0,fc='darkred'))
    # ax.add_patch(Rectangle((-1,-1),2,2,lw=0,fc='darkgreen'))
    # ax.add_patch(Circle((0,0),1,lw=0,fc='g'))
    # ax.add_patch(Ellipse((-3,0),2*1.8,2*2.8,lw=0,fc='r'))
    # ax.add_patch(Rectangle((4-np.sqrt(2)*3/2,4-np.sqrt(2)*3/2),np.sqrt(2)*3,np.sqrt(2)*3))
    # ax.add_patch(Rectangle((-4-np.sqrt(2)*3/2,4-np.sqrt(2)*3/2),np.sqrt(2)*3,np.sqrt(2)*3))
    points = np.array([XX,YY]).T.reshape(-1,1,2)
    segs = np.concatenate([points[:-1],points[1:]],axis=1)
    lc = LineCollection(segs, lw=2, cmap=plt.get_cmap('cividis'))
    lc.set_array(tt)
    ax.add_collection(lc)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel('$p_x$',labelpad=3); ax.set_ylabel('$p_y$',labelpad=3, rotation='horizontal')
    if show_colorbar :
        cb = fig.colorbar(lc, ax=ax, location='right', aspect=40, fraction=0.025, pad=0)
        cb.set_ticks([0,0.25,0.5,0.75,1,1.25])
        cb.set_label('t', rotation='horizontal')
        # cb.set_ticks(list(cb.get_ticks()) + [tt[-1]])
    ax.set_title("y vs x, color t")
def plot_u_t (ax, tt, UU_acc, UU_ang) :
    ax.plot(tt, UU_acc, label='u_acc', color='C4')
    ax2 = ax.twinx()
    ax2.plot(tt, UU_ang, label='u_ang', color='C5')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2)
    ax.set_ylabel('u_acc',labelpad=0)
    ax2.set_ylabel('u_ang',labelpad=0)
    ax.set_title('u_acc,u_ang vs t')
def plot_solution(fig, axs, tt, SS, UU) :
    XX, YY, PP, VV = SS
    UU_acc, UU_ang = UU
    # plot_XY_t(axs[0,0],tt,XX,YY)
    # plot_PV_t(axs[1,0],tt,PP,VV)
    plot_XYPV_t(axs[0,0],tt,SS)
    plot_Y_X (fig,axs[0,1],tt,XX,YY)
    plot_u_t (axs[1,1],tt,UU_acc,UU_ang)