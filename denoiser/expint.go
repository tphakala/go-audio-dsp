package denoiser

import "math"

// expint1 returns the exponential integral E1(x) = int_x^inf e^-t / t dt for
// x > 0, using the Abramowitz & Stegun rational approximations: 5.1.53 for
// x < 1 (absolute error < 2e-7 on E1 + ln x) and 5.1.56 for x >= 1 (relative
// error < 2e-8). Non-positive arguments return +Inf. It is the kernel of the
// MMSE-LSA gain, exp(0.5*E1(v)).
func expint1(x float64) float64 {
	switch {
	case x <= 0:
		return math.Inf(1)
	case x < 1:
		return -math.Log(x) + (((((0.00107857*x-0.00976004)*x+0.05519968)*x-0.24991055)*x+0.99999193)*x - 0.57721566)
	default:
		num := (((x+8.5733287401)*x+18.0590169730)*x+8.6347608925)*x + 0.2677737343
		den := (((x+9.5733223454)*x+25.6329561486)*x+21.0996530827)*x + 3.9584969228
		return num / den * math.Exp(-x) / x
	}
}
