package lang

import (
	"strings"
	"testing"
)

func BenchmarkAnalyze(b *testing.B) {
	for name, text := range map[string]string{
		"english": strings.Repeat("The customer reports that the dashboard will not load after the update. ", 6),
		"german":  strings.Repeat("Ich kann mich nicht in mein Konto einloggen und das Passwort funktioniert nicht. ", 6),
		"chinese": strings.Repeat("我们被收取了两次费用，请退还重复的款项。", 12),
	} {
		b.Run(name, func(b *testing.B) {
			b.SetBytes(int64(len(text)))
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = Analyze(text)
			}
		})
	}
}
