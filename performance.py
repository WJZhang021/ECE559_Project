import numpy as np

def performance_O1(a1, w1=5, q1=30, r1=0.5):
    thres = r1 - (0.4 * w1 * (r1 ** (-0.6))) / (2 * q1)
    f1 = w1 * (thres**0.4) + q1 * (thres - r1)**2
    return np.where(a1 < thres, w1 * a1**0.4, -q1 * (a1 - r1)**2 + f1)
    # if a1 < thres:
    #     return w1 * (a1 ** 0.4)
    # else:
    #     return -q1 * (a1 - r1) ** 2 + f1

def grad_O1(a1, w1=5, q1=30, r1=0.5):
    thres = r1 - (0.4 * w1 * r1**(-0.6)) / (2 * q1)
    return np.where(a1 < thres, w1 * 0.4 * (a1+1e-3)**(-0.6), -2 * q1 * (a1 - r1))
    # if a1 < thres:
    #     return w1 * 0.4 * a1 ** (-0.6)
    # else:
    #     return -2 * q1 * (a1 - r1)



def performance_O2(a2, w2=1, q2=30, r2=0.3, u2=5):
    thres = r2 - ((w2 * np.log(2) * u2 * 2**(u2 * r2)) / (2 * q2))
    f2 = w2 * 2**(u2 * thres) + q2 * (thres - r2)**2
    return np.where(a2 < thres, w2 * 2**(u2 * a2), -q2 * (a2 - r2)**2 + f2)
    # if a2 < thres:
    #     return w2 * 2 ** (u2 * a2)
    # else:
    #     return -q2 * (a2 - r2) ** 2 + f2

def grad_O2(a2, w2=1, q2=30, r2=0.3, u2=5):
    thres = r2 - ((w2 * np.log(2) * u2 * 2**(u2 * r2)) / (2 * q2))
    return np.where(a2 < thres, w2 * np.log(2) * u2 * 2**(u2 * a2), -2 * q2 * (a2 - r2))
    # if a2 < thres:
    #     return w2 * np.log(2) * u2 * 2 ** (u2 * a2)
    # else:
    #     return -2 * q2 * (a2 - r2)



def performance_O3(a3, w3=10, beta3=0.5):
    return w3 * (a3 ** beta3)

def grad_O3(a3, w3=10, beta3=0.5):
    return w3 * beta3 * ((a3+1e-3) ** (beta3 - 1))


def performance_O4(a4, w4=4):
    return w4 * a4


def performance_O5(a5, w5=3):
    return w5 * a5


def synergy_O12(a1, a2, w12=3):
    return w12 * a1 * a2
def grad_O12_a1(a2, w12=3):
    return w12 * a2
def grad_O12_a2(a1, w12=3):
    return w12 * a1


def synergy_O34(a3, a4, w34=0.15, w3=10, beta3=0.5):
    return w34 * performance_O3(a3, w3, beta3) * a4
def grad_O34_a3(a3, a4, w34=0.15, w3=10, beta3=0.5):
    return w34 * grad_O3(a3, w3, beta3) * a4


def synergy_O35(a3, a5, w35=0.5, w3=10, beta3=0.5):
    return w35 * performance_O3(a3, w3, beta3) * a5
def grad_O35_a3(a3, a5, w35=0.5, w3=10, beta3=0.5):
    return w35 * grad_O3(a3, w3, beta3) * a5



def performance_O(a1, a2, a3, a4, a5,
    w1=5, q1=30, r1=0.5, w2=1, q2=30, r2=0.3, u2=5, w3=10,
    beta3=0.5, w4=4, w5=3, w12=3, w34=0.15, w35=0.5):

    return (performance_O1(a1, w1, q1, r1) + performance_O2(a2, w2, q2, r2, u2) +
            performance_O3(a3, w3, beta3) + performance_O4(a4, w4) + performance_O5(a5, w5) +
            synergy_O12(a1, a2, w12) + synergy_O34(a3, a4, w34, w3, beta3) + synergy_O35(a3, a5, w35, w3, beta3))

# just return all the grads
def performance_grad_O(a1, a2, a3, a4, a5,
    w1=5, q1=30, r1=0.5, w2=1, q2=30, r2=0.3, u2=5, w3=10,
    beta3=0.5, w4=4, w5=3, w12=3, w34=0.15, w35=0.5):

    return (grad_O1(a1, w1, q1, r1) + grad_O12_a1(a2, w12),
            grad_O2(a2, w2, q2, r2, u2) + grad_O12_a2(a1, w12),
            grad_O3(a3, w3, beta3) + grad_O34_a3(a3, a4, w34, w3, beta3) + grad_O35_a3(a3, a5, w35, w3, beta3))

