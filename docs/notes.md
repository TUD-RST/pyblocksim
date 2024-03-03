This file contains some notes to facilitate the development process


# Time Discrete Calculations


- time continuous PT1 element: $\frac{K}{1 + T_1 s} = \frac{K/T_1}{s+ 1/T_1}$
- time continuous PT2 element: $\frac{K}{1 + (T_1 + T_2) s + T_1T_2s^2} = \frac{K/(T_1T_2)}{1/(T_1T_2) + (T_1 + T_2)/(T_1 T_2) s + s^2} = \frac{K/(T_1 T_2)}{(s+ 1/T_1)(s+ 1/T_2)}$


### Z - Transfer function

$T$: sampling Time

$$G(z) = (z-1) \sum_\nu \mathrm{Res}\left\{\frac{P(s)}{s}(z-e^{sT})^{-1}\right\}_{s=p_\nu}$$

####  PT1:

$\frac{P(s)}{s}(z-e^{sT})^{-1} = \frac{K/T_1}{s(s+ 1/T_1)}(z-e^{sT})^{-1}, \qquad p_1 = 0; p_2 = -1/T_1 $


$G(z) = (z-1)\cdot\left(K/(z - 1) + - K /(z - e^{-T/T_1})\right) = K \left(1- \frac{z-1}{z - e^{-T/T_1}}\right) =  \frac{K (1 - e^{-T/T_1} )}{z - e^{-T/T_1}} = \frac{Y}{U}$

$Y \cdot (z - e^{-T/T_1})  = U \cdot K (1 - e^{-T/T_1} )$


$y_{k+1} =  u_k \cdot K (1 - e^{-T/T_1}) + y_k e^{-T/T_1}$

#### PT2 ($T_1 \neq T_2$)


$\frac{P(s)}{s}(z-e^{sT})^{-1} = \frac{K/(T_1 T_2)}{s(s+ 1/T_1)(s+ 1/T_2)}, \qquad p_1 = 0; p_2 = -1/T_1, p_3 = -1/T_2 $

$G(z) = K(z-1)\cdot\left(1/(z - 1) +  - 1/T_2*1/(1/T_2 - 1/T_1)*1/(z - e^{-T/T_1}) - 1/T_1*1/(1/T_1 - 1/T_2)*1/(z - e^{-T/T_2})\right)$

$G(z) = K(z-1)\cdot\left(1/(z - 1) - 1/(1-T_2/T_1)*1/(z - e^{-T/T_1}) - 1/(1-T_1/T_2)*1/(z - e^{-T/T_2})\right)$

$G(z) = K(1 - \frac{1}{(1-T_2/T_1)(z - e_1)}- \frac{1}{(1-T_1/T_2)(z - e_2)})$


#### PT2 ($T_1 = T_2$)


$\frac{P(s)}{s}(z-e^{sT})^{-1} = \frac{K/(T_1^2)}{s(s+ 1/T_1)^2}, \qquad p_1 = 0; p_2 = -1/T_1 (mult.=2) $

pole with multiplicity 2 → Res requires taking the derivative

aux calculations:
$a_1:= \frac{d}{ds} \left(\frac{K/T_1^2}{s(s+ 1/T_1)^2} \cdot (z-e^{sT})^{-1}\cdot (s+ 1/T_1)^2\right) = \frac{d}{ds} \left(\frac{K/T_1^2}{s} \cdot (z-e^{sT})^{-1}\right) $

$a_1 = - \frac{K/T_1^2}{s^2}(z-e^{sT})^{-1} + \frac{K/T_1^2}{s} \frac{d}{ds} (z-e^{sT})^{-1}$

$a_2 :=  \frac{d}{ds} (z-e^{sT})^{-1} = -  (z-e^{sT})^{-2}  \frac{d}{ds} (z-e^{sT}) = -  (z-e^{sT})^{-2}  (-Te^{sT})$

→$a_1 = - \frac{K/T_1^2}{s^2}(z-e^{sT})^{-1} + \frac{K/T_1^2}{s} \cdot \frac{Te^{sT}}{(z-e^{sT})^{2}}$

$a_1|_{s = -1/T_1} = -\frac{K}{z-e_1} - \frac{K}{T_1} \frac{Te_1}{(z-e_1)^2} = -\frac{K}{z- e_1}(1 + \frac{T/T_1 e_1}{z-e^{-T/T_1}})$

$G(z) = K(z-1)\cdot\left(1/(z - 1) -\frac{1}{z- e_1}(1 + \frac{T/T_1 e_1}{z-e_1}) \right)$

$G(z) = K(z-1)\cdot\left( \frac{(z-e_1)^2 - (z-1)(z-e_1 + T/T_1 e_1)}{(z -1)(z-e_1)^2} \right) = K\cdot\left( \frac{(z-e_1)^2 - (z-1)(z-e_1 + T/T_1 e_1)}{(z-e_1)^2} \right) = \frac{Y}{U}$


---

dead end:

$Y\cdot(z-e_1)^2 = U\cdot K\cdot ((z-e_1)^2 - (z-1)(z-e_1 + T/T_1 e_1))$


$Y\cdot(z^2-2 z e_1 + e_1 ^2) = U\cdot K\cdot (z^2-2 z e_1 + e_1 ^2 - z^2 + e_1z - T/T_1 e_1z + z-e_1 + T/T_1 e_1)$


$Y\cdot(z^2-2 z e_1 + e_1 ^2) = U\cdot K\cdot ((1-e_1- T/T_1 e_1)z + e_1 ^2   -e_1 + T/T_1 e_1)$

---

canonical realization ( in general)

$G(z) = \frac{a_n z^{-n}+ ... + a_0}{b_n z^{-n}+ ... + 1}$ (note: $b_0 \stackrel{!}{=} 1$)

corresponding difference equation:

$y(k+n) + b_1 y(k + n - 1) + ... + b_n y(k) = a_0 u(k + n) + ... + a_n u(k)$



$y(k+1) + b_1 y(k) + ... + b_n y(k-n + 1) = a_0 u(k + 1) + ... + a_n u(k-n+1)$


$y(k+1) =  - b_1 y(k) - ... - b_n y(k-n + 1) + a_0 u(k + 1) + ... + a_n u(k-n+1)$

---

back to PT2 ($T_1 = T_2$)

$G(z) = K\cdot\left( \frac{(z-e_1)^2 - (z-1)(z-e_1 + T/T_1 e_1)}{(z-e_1)^2} \right)  = K \frac{(z^2-2 z e_1 + e_1 ^2 - z^2 + e_1z - T/T_1 e_1z + z-e_1 + T/T_1 e_1)}{z^2-2 z e_1 + e_1 ^2}$


$G(z) = K \frac{(1- e_1 -  T/T_1 e_1)z + e_1 ^2  -e_1 + T/T_1 e_1}{z^2-2 z e_1 + e_1 ^2} = K \frac{(1- e_1 -  T/T_1 e_1)z^{-1} + (e_1 ^2  -e_1 + T/T_1 e_1)z^{-2}}{1-2 e_1 z^{-1} + e_1 ^2 z^{-2}}$


$b_0 = 1, \quad b_1 = -2 e_1, \quad b_2 = e_1^2 $

$a_0 = 0, \quad a_1 = K(1- e_1 -  T/T_1 e_1), \quad a_2 = K(e_1 ^2  -e_1 + T/T_1 e_1)$


$y(k+1) =  - b_1 y(k) - b_2 y(k- 1) + a_1 u(k)+ ... + a_2 u(k-1)$

##### Implementation

```

x1, x2 = state

u = u(k)

y_new = x1

x2_new = a_2*u - b2*y_new
x1_new = x2 + a_1*u - b1*y_new

```
