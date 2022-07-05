import numpy as np
global D

'''subject to −35 ≤ xi ≤ 35. The global minima is located at origin x∗ = (0, · · · , 0),
f(x∗) = 0.'''
def ackley_1(individuo):
	sum1 = 0.0
	sum2 = 0.0
	for c in individuo:
		sum1 += c**2.0
		sum2 += np.cos(2.0*np.pi*c)
	return -20.0*np.exp(-0.02*np.sqrt(sum1/D)) - np.exp(sum2/D) + 20 + np.e

'''subject to −32 ≤ xi ≤ 32. The global minimum is located at origin x∗ = (0, 0),
f(x∗) = −200. '''  
def ackley_2(individuo):
    return -200*np.exp(-0.02*np.sqrt(individuo[0]**2+individuo[1]**2))

'''subject to −32 ≤ xi ≤ 32. The global minimum is located at x∗ = (0, ≈ −0.4),
f(x∗) ≈ −219.1418.'''
def ackley_3(individuo):
    return -ackley_2(individuo)+5*np.exp(np.cos(3*individuo[0])+np.sin(3*individuo[1]))


'''subject to −35 ≤ xi ≤ 35. It is highly multimodal function with two global minimum
close to origin
x = f({−1.479252, −0.739807}, {1.479252, −0.739807}), f(x∗) = −3.917275'''
def ackley_4(individuo):
    sum = 0.0
    i=0
    while (i+1 < D):
        sum = sum + np.exp(-0.2)*np.sqrt(individuo[i]**2+individuo[i+1]**2)
        +3*(np.cos(2*individuo[i])+np.sin(2*individuo[i+1]))
    return sum