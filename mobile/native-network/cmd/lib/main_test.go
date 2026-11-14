package main

import (
	"sync"
	"testing"

	"nanobot/mobile/native-network/internal/broker"
)

// Requirement (A3-7 AC7.3): concurrent native failures cannot cross request
// identities. Each caller copies its own diagnostic under the library lock and
// must receive one complete message, never a mix, a truncation or an empty
// string.
func TestConcurrentCopiesReturnOneWholeMessage(t *testing.T) {
	errors := []error{
		broker.ErrNotFound,
		broker.ErrQueueFull,
		broker.ErrUnavailable,
		broker.ErrBufferSmall,
	}
	valid := make(map[string]bool, len(errors))
	for _, err := range errors {
		valid[err.Error()] = true
	}
	const workers = 8
	var wg sync.WaitGroup
	start := make(chan struct{})
	failures := make(chan string, workers)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			<-start
			buffer := make([]byte, 128)
			for round := 0; round < 500; round++ {
				setError(errors[(index+round)%len(errors)])
				n := copyLastError(buffer)
				if n < 0 {
					failures <- "copy returned a negative length"
					return
				}
				if buffer[n] != 0 {
					failures <- "copy did not NUL-terminate"
					return
				}
				if !valid[string(buffer[:n])] {
					failures <- "copy returned " + string(buffer[:n])
					return
				}
			}
		}(i)
	}
	close(start)
	wg.Wait()
	close(failures)
	for failure := range failures {
		t.Error(failure)
	}
}

func TestCopyLastErrorTruncatesAndTerminates(t *testing.T) {
	setError(broker.ErrUnavailable)
	buffer := make([]byte, 4)
	n := copyLastError(buffer)
	if n != len(buffer)-1 {
		t.Fatalf("truncated copy length: %d", n)
	}
	if buffer[n] != 0 {
		t.Fatalf("truncated copy is not terminated: %q", buffer)
	}
	if string(buffer[:n]) != "tra" {
		t.Fatalf("truncated copy: %q", buffer[:n])
	}
	if copyLastError(nil) != -1 {
		t.Fatal("empty buffer must be refused")
	}
}

// TestConcurrentErrorWritesAreSynchronized is the pre-repair reproducer: the
// unsynchronized slot made `go test -race` fail. It stays as the race guard for
// the legacy library-owned pointer.
func TestConcurrentErrorWritesAreSynchronized(t *testing.T) {
	errors := []error{
		broker.ErrNotFound,
		broker.ErrQueueFull,
		broker.ErrUnavailable,
		broker.ErrBufferSmall,
	}
	const workers = 8
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			<-start
			for round := 0; round < 500; round++ {
				setError(errors[(index+round)%len(errors)])
				_ = nanobot_network_last_error()
			}
		}(i)
	}
	close(start)
	wg.Wait()
}
