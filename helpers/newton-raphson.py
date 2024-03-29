import numpy as np

def newtonRaphson(f,x0,e,N):
    print('\n\n*** NEWTON RAPHSON METHOD IMPLEMENTATION ***')
    step = 1
    flag = 1
    condition = True
    while condition:
        fx0, dfx0 = f(x0)
        if dfx0 == 0.0:
            print('Divide by zero error!')
            break
        elif np.linalg.norm(fx0) <= e:
            print('Converged!')
        
        x1 = x0 - fx0/dfx0
        print('Iteration-%d, x1 = %0.6f' % (step, x1))
        x0 = x1
        step = step + 1
        
        if step > N:
            flag = 0
            break
        
        condition = abs(f(x1)) > e
    
    if flag==0:
        print('\nNot Convergent.')
    return x0


nu = 7.415414466563401e-05 # Re ~ 4e5
hwm = 0.3070183607238605
deltax = 0.04605275410857908
kappa = 0.41
A = 10.

def g(ut):
    val = u_wm/ut - 1/kappa*numpy.log(hwm*ut/nu)-A
    valprime = -1/ut * (u_wm/ut - 1/kappa)
    return val, numpy.diag(valprime)


