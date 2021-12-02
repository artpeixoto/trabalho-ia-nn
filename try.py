import tensorflow as tf
import logging 

neural_layer = lambda fn: (
    lambda parms: ( 
        lambda x:  (
                fn(parms @ x) #note que com uma definição funcional como essa, qualquer implementacao das operacoes basicas pode ser usada, desde que se atenham a convencao da linguagem (foo.__matmul__).
        )
    )
)

def neural_net_function(parms, activation_fns):
    first_layer, second_layer = (neural_layer(activation_fn)(parm) for parm, activation_fn in zip(parms, activation_fns))
    return (lambda input: second_layer(first_layer(input)))

#terminamos nossas definicoes basicas. agora, vamos comecar nossas experimentacoes

DTYPE = tf.float64 #para garantir a precisão requerida, usamos o maior float disponivel

DATA_SIZES = (3,10,)





ACTIVATION_FUNCTION = tf.sigmoid

FIRS




#criamos um pequeno ajudante para vermos o que esta acontecendo
def verboiser(message_function, original_function):
    def wrapper(*args, **kwargs):
        res =  original_function(*args, **kwargs)
        message_function(res, *args, **kwargs,)
        return res
    return wrapper

if debug: ACTIVATION_FUNCTION = verboiser(
    (lambda res, *args, **kwargs: 
        print(f"running layer with the following args: {args}; {kwargs}"))
    , ACTIVATION_FUNCTION)


