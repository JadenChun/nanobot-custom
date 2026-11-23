package main

import (
	"runtime"
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

// Requirement case C2 (independent requirements U2/R2): two concurrent callers
// that fail must each copy their own error text. Two goroutines are locked to
// distinct OS threads and yield between the failing call and the copy, which is
// the interleaving a single process-wide slot cannot survive.
func TestConcurrentGoroutinesKeepTheirOwnError(t *testing.T) {
	errA, errB := broker.ErrNotFound, broker.ErrInvalid
	if errA.Error() == errB.Error() {
		t.Fatal("fixture errors must have distinct text")
	}
	var wg sync.WaitGroup
	start := make(chan struct{})
	failures := make(chan string, 2)
	run := func(err error) {
		defer wg.Done()
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		<-start
		buffer := make([]byte, 256)
		for round := 0; round < 2000; round++ {
			setError(err)
			runtime.Gosched()
			n := copyLastError(buffer)
			if n < 0 || string(buffer[:n]) != err.Error() {
				failures <- "own=" + err.Error() + " got=" + string(buffer[:n])
				return
			}
		}
	}
	wg.Add(2)
	go run(errA)
	go run(errB)
	close(start)
	wg.Wait()
	close(failures)
	for failure := range failures {
		t.Error(failure)
	}
}

// The store keeps one message per thread and stays bounded. This is the
// deterministic form of C2: the interleaving is explicit, not scheduled.
func TestThreadErrorStoreKeepsIdentity(t *testing.T) {
	errA, errB := broker.ErrNotFound, broker.ErrInvalid
	buffer := make([]byte, 256)
	for round := 0; round < 2000; round++ {
		recordError(101, errA)
		recordError(102, errB)
		if n := copyErrorForThread(101, buffer); string(buffer[:n]) != errA.Error() {
			t.Fatalf("round %d: thread 101 read %q", round, buffer[:n])
		}
		if n := copyErrorForThread(102, buffer); string(buffer[:n]) != errB.Error() {
			t.Fatalf("round %d: thread 102 read %q", round, buffer[:n])
		}
	}
	for i := 0; i < maxThreadErrors+10; i++ {
		recordError(int32(1000+i), broker.ErrQueueFull)
	}
	lastErrorMu.Lock()
	size := len(lastErrors)
	lastErrorMu.Unlock()
	if size > maxThreadErrors {
		t.Fatalf("thread error store holds %d entries, limit %d", size, maxThreadErrors)
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
