package denoiser

// NoiseProfile and mcra are placeholders for noise-profile and adaptive-tracker
// support that is not implemented yet; these minimal empty forms let the
// streaming core compile and satisfy the unused-field linter.
type NoiseProfile struct{}

type mcra struct{}

func (m *mcra) reset()                 {}
func (m *mcra) update(power []float32) {}
