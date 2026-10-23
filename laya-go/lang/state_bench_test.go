package lang

import (
	"fmt"
	"testing"
)

func manyLeaves(n int) []any {
	leaves := make([]any, n)
	for i := range leaves {
		leaves[i] = fmt.Sprintf("The customer reports problem number %d with the dashboard", i)
	}
	return leaves
}

// StateText stops reading once its 4000-character budget is spent, so a huge state costs the
// same as a small one; Analyze must read every leaf, so it grows with the number of leaves.
func BenchmarkStateText(b *testing.B) {
	for _, n := range []int{100, 100_000} {
		state := manyLeaves(n)
		b.Run(fmt.Sprintf("%d-leaves", n), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = StateText(state)
			}
		})
	}
}

func BenchmarkAnalyzeLeaves(b *testing.B) {
	for _, n := range []int{100, 100_000} {
		state := manyLeaves(n)
		b.Run(fmt.Sprintf("%d-leaves", n), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_ = Analyze(state)
			}
		})
	}
}
