package model

import (
	"fmt"
	"sort"
	"sync"
)

var (
	registryMu sync.RWMutex
	registry   = map[string]func() Persistable{}
)

// Register makes an algorithm known to the loader. Algorithm packages call this
// from an init function, so that importing the package (even with a blank
// import) is enough for Load to reconstruct its models:
//
//	import _ "github.com/nakurai/goblas-ai/linear"
//
// factory must return a fresh, zero-valued instance ready to receive
// UnmarshalWeights. Register panics if the same name is registered twice, which
// indicates a programming error.
func Register(algo string, factory func() Persistable) {
	registryMu.Lock()
	defer registryMu.Unlock()
	if _, exists := registry[algo]; exists {
		panic(fmt.Sprintf("model: algorithm %q registered twice", algo))
	}
	registry[algo] = factory
}

// lookup returns the factory for algo, or an error naming the algorithms that
// are registered (a common cause of failure is forgetting the blank import).
func lookup(algo string) (func() Persistable, error) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	factory, ok := registry[algo]
	if !ok {
		return nil, fmt.Errorf("model: unknown algorithm %q; registered: %v "+
			"(did you forget to import its package?)", algo, registeredNames())
	}
	return factory, nil
}

// registeredNames returns the sorted list of registered algorithm names.
func registeredNames() []string {
	names := make([]string, 0, len(registry))
	for name := range registry {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
