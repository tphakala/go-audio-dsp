package audiotest

// AfftdnPreset maps a denoiser strength name (denoiser.Strength.String()) to the
// BirdNET-Go afftdn parameters (nr dB, nf dBFS): the exact reduction and
// noise-floor values production uses, so an A/B measures the baseline being
// replaced, not a re-tuned afftdn. A strength whose nf sits well below a clip's
// noise level (heavy at nf=-50 against -40 dBFS synthetic noise) leaves afftdn
// barely engaged, so both A/B bars are easy to clear for it: afftdn reduces
// little and distorts the signal little. The definitive comparison runs on real
// recordings, where noise floors match production.
var AfftdnPreset = map[string][2]int{
	"light":  {6, -30},
	"medium": {12, -40},
	"heavy":  {20, -50},
}
