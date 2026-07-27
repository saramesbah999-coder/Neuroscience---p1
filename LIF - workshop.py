import numpy as np
import matplotlib.pyplot as plt

t_max = 150e-3   # second
dt = 1e-3        # second
tau = 20e-3      # second
el = -60e-3      # milivolt
vr = -70e-3      # milivolt
vth = -50e-3     # milivolt
r = 100e6        # ohm
i_mean = 25e-11  #ampere 
v = el
for step in range (20):
    t = step * dt
    i = i_mean * (1 + np.sin((t * 2 * np.pi) / 0.01))
    v = v + dt/tau * (el - v + r*i)
    print(f"{t:.3f} {v:.4e}")
    plt.plot(t, i, "k2")
plt.title("LIF model")
plt.xlabel("Time (s)")
plt.ylabel("I (amp)")
plt.show()

#with randomization

np.random.seed(2030)
step_end = int(t_max / dt)
n = 50

v_n = [el]*n

with plt.xkcd():
  
  plt.figure()
  plt.title('Multiple realizations of $V_m$')
  plt.xlabel('time (s)')
  plt.ylabel('$V_m$ (V)')

  for step in range(step_end):
    t = step * dt
    for j in range(0, n):
      i = i_mean * (1 + 0.1 * (t_max/dt)**(0.5) * (2* np.random.random() - 1))

      v_n[j] = v_n[j] + (dt / tau) * (el - v_n[j] + r*i)

    
    plt.plot([t] * n, v_n, 'k.', alpha=0.1)
  plt.show()