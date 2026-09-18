package mel

import "testing"

func TestEnumStrings(t *testing.T) {
	cases := []struct {
		got  string
		want string
	}{
		{Slaney.String(), "slaney"},
		{HTK.String(), "htk"},
		{MelScale(9).String(), "MelScale(9)"},
		{NormSlaney.String(), "slaney"},
		{NormNone.String(), "none"},
		{Norm(9).String(), "Norm(9)"},
		{InputPower.String(), "power"},
		{InputMagnitude.String(), "magnitude"},
		{Input(9).String(), "Input(9)"},
		{LogNone.String(), "none"},
		{Log10.String(), "log10"},
		{LogNatural.String(), "natural"},
		{Log(9).String(), "Log(9)"},
	}
	for _, tc := range cases {
		if tc.got != tc.want {
			t.Errorf("String() = %q, want %q", tc.got, tc.want)
		}
	}
}
