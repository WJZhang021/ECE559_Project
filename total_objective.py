from performance import performance_O, performance_grad_O
from costs import total_C, cost_grad_C

def TO(a1, a2, a3, a4, a5,
                    B=30, lambda_val=0.3,
                    w1=5, q1=30, r1=0.5, w2=1, q2=30, r2=0.3, u2=5, w3=10,
                    beta3=0.5, w4=4, w5=3, w12=3, w34=0.15, w35=0.5,
                    c1=8, c2=0.001, c3=10, alpha3=2, s3=5, c4=5, c5=10):
    # Calculate total performance
    O = performance_O(a1, a2, a3, a4, a5, w1, q1, r1, w2, q2, r2, u2, w3, beta3, w4, w5, w12, w34, w35)
    
    # Calculate total cost
    C = total_C(a1, a2, a3, a4, a5, c1, c2, u2, c3, alpha3, s3, c4, c5)
    
    # Calculate total objective
    if C <= 0.75 * B:
        return O
    else:
        return O - lambda_val * (C - 0.75 * B) ** 2

def grad_TO_O():
    return 1

def grad_TO_grad_C(C, B=30, lambda_val=0.3):
    if C <= 0.75 * B:
        return 0
    else:
        return -2 * lambda_val * (C - 0.75 * B)


def grad_TO_grad_ai(a1, a2, a3, a4, a5,
                    B=30, lambda_val=0.3, w1=5, q1=30, r1=0.5, w2=1, q2=30, r2=0.3, u2=5, w3=10,
                    beta3=0.5, w4=4, w5=3, w12=3, w34=0.15, w35=0.5, c1=8, c2=0.001, c3=10,
                    alpha3=2, s3=5, c4=5, c5=10):
    # Calculate total cost
    C = total_C(a1, a2, a3, a4, a5, c1, c2, u2, c3, alpha3, s3, c4, c5)
    
    # Calculate partial derivatives of total performance
    grad_O = performance_grad_O(a1, a2, a3, a4, a5, w1, q1, r1, w2, q2, r2, u2, w3, beta3, w4, w5, w12, w34, w35)
    
    # Calculate partial derivatives of total cost
    grad_C = cost_grad_C(a1, a2, a3, c1, c2, u2, c3, alpha3)
    
    # Calculate partial derivatives of total objective
    if C <= 0.75 * B:
        return grad_O
    else:
        grad_TO_C = grad_TO_grad_C(C, B, lambda_val)
        return (grad_O[0] + grad_TO_C * grad_C[0],
                grad_O[1] + grad_TO_C * grad_C[1],
                grad_O[2] + grad_TO_C * grad_C[2])