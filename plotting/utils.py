from enum import Enum
import seaborn as sns
import matplotlib as mpl
import matplotlib.axes as axes

class Color(Enum):
    LIGHT_GREY = (240 / 256, 240 / 256, 240 / 256)


def style_plot(ax: axes.Axes):
    mpl.rcParams['legend.loc'] = "upper right"
    mpl.rc('legend',fontsize='smaller')
    ax.grid(True, color=Color.LIGHT_GREY.value)
    sns.despine(left=True, bottom=True, right=True, top=True)
    #mpl.rc('font',**{'family':'IBM Plex Sans'})
    #mpl.rc('text', usetex=True)
