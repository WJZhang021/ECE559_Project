import numpy as np
import matplotlib.pyplot as plt
from costs import cost_C1, cost_C2, cost_C3, cost_C4, cost_C5
from performance import performance_O1, performance_O2, performance_O3, performance_O4, performance_O5, synergy_O12, synergy_O34, synergy_O35

# Generate the range of independent variable a [0, 1]
a = np.linspace(0, 1, 1000)

# Plotting the Cost functions
plt.figure()
plt.xlabel('a_i')
plt.ylabel('C')
plt.xlim([0, 1])
plt.ylim([0, 20])
plt.grid(True)

# Plot C1
C1 = cost_C1(a)
plt.plot(a, C1, 'g-', linewidth=1.5, label='C_1')

# Plot C2
C2 = cost_C2(a)
plt.plot(a, C2, 'b-', linewidth=1.5, label='C_2')

# Plot C3
C3 = cost_C3(a)
plt.plot(a, C3, 'r-', linewidth=1.5, label='C_3')

# Plot C4
C4 = cost_C4(a>0)
plt.plot(a, C4, 'm:', linewidth=1.5, label='C_4')

# Plot C5
C5 = cost_C5(a>0)
plt.plot(a, C5, 'c:', linewidth=1.5, label='C_5')

plt.title('Cost')
plt.legend()
plt.show()

# Plotting the Performance functions
plt.figure()
plt.xlabel('a_i')
plt.ylabel('O')
plt.xlim([0, 1])
plt.ylim([-15, 15])
plt.grid(True)

# Plot O1
O1 = performance_O1(a)
plt.plot(a, O1, 'g-', linewidth=1.5, label='O_1')

# Plot O2
O2 = performance_O2(a)
plt.plot(a, O2, 'b-', linewidth=1.5, label='O_2')

# Plot O3
O3 = performance_O3(a)
plt.plot(a, O3, 'r--', linewidth=1.5, label='O_3')

# Plot O4
O4 = performance_O4(a>0)
plt.plot(a, O4, 'm:', linewidth=1.5, label='O_4')

# Plot O5
O5 = performance_O5(a>0)
plt.plot(a, O5, 'c:', linewidth=1.5, label='O_5')

# # Plot Synergy (1,2)
# O12 = synergy_O12(a, a)
# plt.plot(a, O12, 'y-', linewidth=1.5, label='Syn(1,2)')

# # Plot Synergy (3,4)
# O34 = synergy_O34(a, a)
# plt.plot(a, O34, 'k-', linewidth=1.5, label='Syn(3,4)')

# # Plot Synergy (3,5)
# O35 = synergy_O35(a, a)
# plt.plot(a, O35, 'b:', linewidth=1.5, label='Syn(3,5)')

# Plot Synergy (3,5)
O35 = synergy_O35(a, 1) + performance_O3(a)
plt.plot(a, O35, 'b:', linewidth=1.5, label='O3+Syn(3,5)')

plt.title('Performance')
plt.legend()
plt.show()