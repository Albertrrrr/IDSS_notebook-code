import IPython.display
import contextlib
from contextlib import contextmanager

@contextmanager
def marks(marks):
    try:
        yield
        IPython.display.display(IPython.display.HTML('<h3> <font color="green"> ✓ [%d marks] </font> </h3>' % marks))    
    except Exception as e:    
        IPython.display.display(IPython.display.HTML('<hr style="height:10px;border:none;color:#f00;background-color:#f00;" /><h3> <font color="red"> Test failed ✘ [0/%d] marks </font> </h3>' % marks))        
        raise e
    
    