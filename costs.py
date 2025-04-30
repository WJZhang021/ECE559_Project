import numpy as np

def cost_C1(a1, c1=8):
    return c1 * a1
def grad_C1(a1, c1=8):
    return c1


def cost_C2(a2, c2=0.001, u2=5):
    return c2 * 10 ** (u2 * a2)
def grad_C2(a2, c2=0.001, u2=5):
    return c2 * np.log(10) * u2 * (10 ** (u2 * a2))


def cost_C3(a3, c3=10, alpha3=2, s3=5):
    return (c3 * (a3 ** alpha3)) + s3
def grad_C3(a3, c3=10, alpha3=2):
    return c3 * alpha3 * (a3 ** (alpha3 - 1))


def cost_C4(a4, c4=5):
    return c4 * a4


def cost_C5(a5, c5=10):
    return c5 * a5


def total_C(a1, a2, a3, a4, a5, c1=8, c2=0.001, u2=5, c3=10, alpha3=2, s3=5, c4=5, c5=10):
    return (cost_C1(a1, c1) + cost_C2(a2, c2, u2) + 
            cost_C3(a3, c3, alpha3, s3) + cost_C4(a4, c4) + cost_C5(a5, c5))

# just return all the grads
def cost_grad_C(a1, a2, a3, c1=8, c2=0.001, u2=5, c3=10, alpha3=2):
    return (grad_C1(a1, c1), grad_C2(a2, c2, u2), grad_C3(a3, c3, alpha3))