import numpy as np
from total_objective import TO, grad_TO_grad_ai
import scipy.io as sio

def gradient_ascent(a1_init, a2_init, a3_init, a4_init, a5_init,
                    B=30, lambda_val=0.3, w1=5, q1=30, r1=0.5, w2=1,
                    q2=30, r2=0.3, u2=5, w3=10, beta3=0.5, w4=4, w5=3, w12=3, w34=0.15, w35=0.5,
                    c1=8, c2=0.001, c3=10, alpha3=2, s3=5, c4=5, c5=10, lr=1e-3, max_iter=1000, tol=1e-6):
    
    # Initialize power allocation variables
    a1, a2, a3, a4, a5 = a1_init, a2_init, a3_init, a4_init, a5_init
    obj_list = np.empty(max_iter+1)
    obj_list[0] = TO(a1, a2, a3, a4, a5, B, lambda_val, w1, q1, r1, w2, q2, r2, u2, w3, beta3, w4, w5, w12, w34, w35, c1, c2, c3, alpha3, s3, c4, c5)
        
    for i in range(max_iter):
        # Calculate the gradient of the total objective function
        grad_a1, grad_a2, grad_a3  = grad_TO_grad_ai(a1, a2, a3, a4, a5, B, lambda_val, w1, q1, r1, w2, q2, r2, u2, w3, beta3, w4, w5, w12, w34, w35, c1, c2, c3, alpha3, s3, c4, c5)
        
        # Update power allocation variables
        a1 = a1 + lr * grad_a1
        a2 = a2 + lr * grad_a2
        a3 = a3 + lr * grad_a3
        
        # Project variables to the valid range [0, 1] or {0, 1}
        a1 = np.clip(a1, 0, 1)
        a2 = np.clip(a2, 0, 1)
        a3 = np.clip(a3, 0, 1)
        
        obj_list[i+1] = TO(a1, a2, a3, a4, a5, B, lambda_val, w1, q1, r1, w2, q2, r2, u2, w3, beta3, w4, w5, w12, w34, w35, c1, c2, c3, alpha3, s3, c4, c5)
        
        # Check for convergence
        # condition = np.linalg.norm(np.array([a1_new, a2_new, a3_new]) - np.array([a1, a2, a3]))
        # if condition < tol:
        #     break
        #print(f"Condition:{condition}, Objective: {objective} ")
        
        if np.abs(obj_list[i+1]-obj_list[i])/obj_list[i]<tol:
            break               
    
    return (a1, a2, a3, a4, a5), obj_list[:i+2], i

def tree_search(B=30, lambda_val=0.3, w1=5, q1=30, r1=0.5, w2=1,
                    q2=30, r2=0.3, u2=5, w3=10, beta3=0.5, w4=4, w5=3, w12=3, w34=0.15, w35=0.5,
                    c1=8, c2=0.001, c3=10, alpha3=2, s3=5, c4=5, c5=10, lr=1e-3, max_iter=1000, tol=1e-6):
    discrete = [0,1]
    contiuous_init = [(0,0,0), (0.5,0.5,0.5), (1,1,1)]
    
    obj_final4 = np.empty(4)
    allo4 = []
    obj_list4 = []
    iter4 = np.empty(4)
    
    ind =0
    ind_op = 0
    for a4 in discrete:
        for a5 in discrete:
            obj_final_op = -1e10
            allo_op = None
            obj_list_op = None
            iter_op = 0            
            for (a1_init, a2_init, a3_init) in contiuous_init:
                allo, obj_list, i = gradient_ascent(a1_init, a2_init, a3_init, a4, a5,
                    B, lambda_val, w1, q1, r1, w2, q2, r2, u2, w3, beta3, w4, w5, w12, w34, w35,
                    c1, c2, c3, alpha3, s3, c4, c5, lr, max_iter, tol)
                if obj_list[-1] > obj_final_op:
                    obj_final_op = obj_list[-1]
                    allo_op = allo
                    obj_list_op = obj_list
                    iter_op = i
            obj_final4[ind] = obj_final_op
            allo4.append(allo_op)
            obj_list4.append(obj_list_op)
            iter4[ind] = iter_op
            ind += 1
            
    ind_op = np.argmax(obj_final4)
    return allo4, obj_final4, obj_list4, iter4, ind_op


if __name__ == "__main__":   
    para = {
        'B': 30, 'lambda_val': 0.3,
        'w1': 5, 'q1': 30, 'r1': 0.5,
        'w2': 1, 'q2': 30, 'r2': 0.3, 'u2': 5,
        'w3': 10, 'beta3': 0.5,
        'w4': 4, 'w5': 3,
        'w12': 3, 'w34': 0.15, 'w35': 0.5,
        'c1': 8, 'c2': 0.001, 'c3': 10, 'alpha3': 2, 's3': 5,
        'c4': 5, 'c5': 10,
        'lr': 1e-3, 'max_iter': 1000, 'tol': 1e-6
    }
    allo4, obj_final4, obj_list4, iter4, ind_op = tree_search(**para)

    output_data = {
            "para": para,
            "allo4": allo4,
            "obj_final4": obj_final4,
            "obj_list4": obj_list4,
            "iter4": iter4,
            "ind_op": ind_op 
        }
        
    # 保存结果
    save_dir = 'Results/Apr29.mat'
    sio.savemat(save_dir, output_data)

    print(f"Results saved to {save_dir}")
    
    print(f"Optimal Allocation: {allo4[ind_op]}")
    print(f"Optimal Objective Value: {obj_final4[ind_op]}")
    print(f"Optiaml Iterations: {iter4[ind_op]}")
    