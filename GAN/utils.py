from traitlets.config.configurable import Configurable
from traitlets.config.application import  Application

from traitlets import Unicode, Int, Float, List, Dict, Tuple, Bool

from IPython.core.getipython import get_ipython

import os

def param(val):
    if type(val) == str:
        return Unicode(val).tag(config=True)
    elif type(val) == int:
        return Int(val).tag(config=True)
    elif type(val) == bool:
        return Bool(val).tag(config=True)
    elif type(val) == float:
        return Float(val).tag(config=True)
    elif type(val) == list:
        return List(val).tag(config=True)
    elif type(val) == tuple:
        return Tuple(val).tag(config=True)
    elif type(val) == dict:
        return Dict(val).tag(config=True)                                
    raise Exception("Unhandled type %s" % type(val) )


class Parameters(Configurable):
    
    def __init__(self,myapp,**kwargs):
        
        self._params = list(filter(lambda x: not x.startswith("_"), self.__class__.__dict__.keys() ))
        super(Parameters,self).__init__(config=myapp.config)

    def get_params(self):
        return { name.upper():getattr(self,name) for name in self._params }

        
class MyApp(Application):

    config_file = Unicode(u'', 
                          help="Load this config file").tag(config=True)
    
    def __init__(self):

        super(MyApp,self).__init__()

        my_app_args=os.environ.get("MY_APP_ARGS","").split(" ")
        self.parse_command_line(my_app_args)
        if self.config_file:
            self.load_config_file(self.config_file)
        