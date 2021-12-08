import logging 
import sys
#---------------------------------------------------
# um pouco de debug

logging_level = logging.INFO

#criamos um pequeno ajudante para xeretar alguma funcao qualquer.

def verboiser(message_function, original_function):
        def wrapper(*args, **kwargs):
            res =  original_function(*args, **kwargs)
            message_function(res, *args, **kwargs,)
            return res
        return wrapper


rlogger = logging.RootLogger(logging_level)
logger = logging.getLogger()
logger.setLevel(logging_level)
logger.addHandler(logging.StreamHandler(sys.stdout))
#---------------------------------------------------
