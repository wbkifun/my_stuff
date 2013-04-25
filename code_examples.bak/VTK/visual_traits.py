from numpy import linspace, pi, cos, sin
from enthought.traits.api import HasTraits, Range, Instance, on_trait_change
from enthought.traits.ui.api import View, Item, HGroup
from enthought.mayavi.core.ui.api import SceneEditor, MlabSceneModel


def curve(n_turns):
    phi = linspace(0, 2*pi, 2000)
    return [ cos(phi) * (1 + 0.5*cos(n_turns*phi)),
             sin(phi) * (1 + 0.5*cos(n_turns*phi)),
             0.5*sin(n_turns*phi)]


class Visualization(HasTraits):
    n_turns = Range(0, 30, 11)
    scene = Instance(MlabSceneModel, ())


    def __init__(self):
        HasTraits.__init__(self)
        x, y, z = curve(self.n_turns) 
        self.plot = self.scene.mlab.plot3d(x, y, z)


    @on_trait_change('n_turns')
    def update_plot(self):
        x, y, z = curve(self.n_turns)
        self.plot.mlab_source.set(x=x, y=y, z=z)
        view = View(Item('scene', height=300, show_label=False, 
                    editor=SceneEditor()),
                    HGroup('n_turns'), resizable=True)


Visualization().configure_traits()
