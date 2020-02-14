#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import matplotlib.pyplot as plt

def model_A(x, params):
    y = params[0]+x*params[1]+params[2]*x**2
    return y
    
def model_B(x,params):
    y=params[0]+(np.exp(-0.5*(x-params[1])**2/params[2]**2))
    return y

def model_C(x,params):
    y = params[0]*(np.exp(-0.5*(x-params[1])**2/params[2]**2))
    y += params[0]*(np.exp(-0.5*(x-params[3])**2/params[4]**2))
    return y
    
def loglike_A(x_obs, y_obs, sigma_y_obs, params):
    n_obs = len(y_obs)
    l = 0.0
    for i in range(n_obs):
        l += -0.5*(y_obs[i]-model_A(x_obs[i], params))**2/sigma_y_obs[i]**2
    return l

def loglike_B(x_obs, y_obs, sigma_y_obs, params):
    n_obs = len(y_obs)
    l = 0.0
    for i in range(n_obs):
        l += -0.5*(y_obs[i]-model_B(x_obs[i], params))**2/sigma_y_obs[i]**2
    return l

def loglike_C(x_obs, y_obs, sigma_y_obs, params):
    n_obs = len(y_obs)
    l = 0.0
    for i in range(n_obs):
        l += -0.5*(y_obs[i]-model_C(x_obs[i], params))**2/sigma_y_obs[i]**2
    return l

def mcmc_A(data_file="data_to_fit.txt", n_dim=2, n_iterations=10000):
    data = np.loadtxt(data_file,skiprows=1)
    x_obs = data[:,0]
    y_obs = data[:,1]
    sigma_y_obs = data[:,2]
    
    params = np.zeros([n_iterations, n_dim+1])
    for i in range(1, n_iterations):
        current_params = params[i-1,:]
        next_params = current_params + np.random.normal(scale=0.01, size=n_dim+1)

        loglike_current = loglike_A(x_obs, y_obs, sigma_y_obs, current_params)
        loglike_next = loglike_A(x_obs, y_obs, sigma_y_obs, next_params)

        r = np.min([np.exp(loglike_next - loglike_current), 1.0])
        alpha = np.random.random()

        if alpha < r:
            params[i,:] = next_params
        else:
            params[i,:] = current_params
    params = params[n_iterations//2:,:]
    return {'parameters':params, 'x_obs':x_obs, 'y_obs':y_obs}

def mcmc_B(data_file="data_to_fit.txt", n_dim=2, n_iterations=10000):
    data = np.loadtxt(data_file)
    x_obs = data[:,0]
    y_obs = data[:,1]
    sigma_y_obs = data[:,2]

    params = np.zeros([n_iterations, n_dim+1])
    for i in range(1, n_iterations):
        current_params = params[i-1,:]
        next_params = current_params + np.random.normal(scale=0.01, size=n_dim+1)

        loglike_current = loglike_B(x_obs, y_obs, sigma_y_obs, current_params)
        loglike_next = loglike_B(x_obs, y_obs, sigma_y_obs, next_params)

        r = np.min([np.exp(loglike_next - loglike_current), 1.0])
        alpha = np.random.random()

        if alpha < r:
            params[i,:] = next_params
        else:
            params[i,:] = current_params
    params = params[n_iterations//2:,:]
    return {'parameters':params, 'x_obs':x_obs, 'y_obs':y_obs}

def mcmc_C(data_file="data_to_fit.txt", n_dim=4, n_iterations=10000):
    data = np.loadtxt(data_file)
    x_obs = data[:,0]
    y_obs = data[:,1]
    sigma_y_obs = data[:,2]

    params = np.zeros([n_iterations, n_dim+1])
    for i in range(1, n_iterations):
        current_params = params[i-1,:]
        next_params = current_params + np.random.normal(scale=0.01, size=n_dim+1)

        loglike_current = loglike_C(x_obs, y_obs, sigma_y_obs, current_params)
        loglike_next = loglike_C(x_obs, y_obs, sigma_y_obs, next_params)

        r = np.min([np.exp(loglike_next - loglike_current), 1.0])
        alpha = np.random.random()

        if alpha < r:
            params[i,:] = next_params
        else:
            params[i,:] = current_params
    params = params[n_iterations//2:,:]
    return {'parameters':params, 'x_obs':x_obs, 'y_obs':y_obs}


# In[77]:


data=np.loadtxt('data_to_fit.txt')
n_dim = 2
results = mcmc_A()
params = results['parameters']
plt.figure(figsize=((8,8)))
for i in range(0,n_dim+1):
    plt.subplot(2,2,i+1)
    plt.hist(params[:,i],bins=15, density=True)
    plt.title(r"$p_{}= {:.2f}\pm {:.2f}$".format(i,np.mean(params[:,i]), np.std(params[:,i])))
    plt.xlabel(r"$p_{}$".format(i))
plt.subplots_adjust(hspace=0.65)
plt.subplot(2,2,4)
plt.errorbar(results['x_obs'],results['y_obs'],yerr=data[:,2], marker='.',linestyle='None')
x_obs=data[:,0]
y_test=np.empty((0))
x_test=np.empty((0))
for i in range(len(x_obs)):
    y_test=np.append(y_test,model_A(x_obs[i],params[i]))
    x_test=np.append(x_test,x_obs[i])

plt.plot(x_test,y_test,'-o',linestyle='None')



#plt.savefig("modelo_A.png",  bbox_inches='tight')    


# In[80]:


n_dim = 2
results = mcmc_B()
params = results['parameters']
plt.figure()
for i in range(0,n_dim+1):
    plt.subplot(2,2,i+1)
    plt.hist(params[:,i],bins=15, density=True)
    plt.title(r"$p_{}= {:.2f}\pm {:.2f}$".format(i,np.mean(params[:,i]), np.std(params[:,i])))
    plt.xlabel(r"$p_{}$".format(i))
plt.subplots_adjust(hspace=0.65)

plt.subplot(2,2,4)
plt.errorbar(results['x_obs'],results['y_obs'],yerr=data[:,2], marker='.',linestyle='None')
x_obs=data[:,0]
y_test=np.empty((0))
for i in range(len(x_obs)):
    y_test=np.append(y_test,model_B(x_obs[i],params[i]))

plt.plot(x_obs,y_test,'-o',linestyle='None')

plt.savefig("modelo_B.png",  bbox_inches='tight')    


# In[82]:


n_dim = 4
results = mcmc_C()
params = results['parameters']
plt.figure()
for i in range(0,n_dim+1):
    plt.subplot(2,3,i+1)
    plt.hist(params[:,i],bins=15, density=True)
    plt.title(r"$p_{}= {:.2f}\pm {:.2f}$".format(i,np.mean(params[:,i]), np.std(params[:,i])))
    plt.xlabel(r"$p_{}$".format(i))
plt.subplots_adjust(hspace=0.65)

plt.subplot(2,2,4)
plt.errorbar(results['x_obs'],results['y_obs'],yerr=data[:,2], marker='.',linestyle='None')
x_obs=data[:,0]
y_test=np.empty((0))
for i in range(len(x_obs)):
    y_test=np.append(y_test,model_C(x_obs[i],params[i]))

plt.plot(x_obs,y_test,'-o',linestyle='None')



plt.savefig("modelo_C.png",  bbox_inches='tight')    


# In[ ]:




