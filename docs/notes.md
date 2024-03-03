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
