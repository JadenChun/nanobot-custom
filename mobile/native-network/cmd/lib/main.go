package main

/*
#include <stdint.h>
#include <stdlib.h>
*/
import "C"

import (
	"encoding/json"
	"errors"
	"sync"
	"unsafe"

	"nanobot/mobile/native-network/internal/broker"
)

var (
	managerMu  sync.Mutex
	cores      = map[uint64]*broker.Core{}
	nextHandle uint64
	lastErrC   *C.char
	lastErr    string
)

func main() {}

func setError(err error) C.int32_t {
	lastErr = err.Error()
	if lastErrC != nil {
		C.free(unsafe.Pointer(lastErrC))
	}
	lastErrC = C.CString(lastErr)
	return codeFor(err)
}

func codeFor(err error) C.int32_t {
	switch {
	case err == nil:
		return 0
	case errors.Is(err, broker.ErrInvalid):
		return -1
	case errors.Is(err, broker.ErrDenied):
		return -2
	case errors.Is(err, broker.ErrBusy):
		return -3
	case errors.Is(err, broker.ErrNotFound):
		return -4
	case errors.Is(err, broker.ErrQueueFull):
		return -5
	case errors.Is(err, broker.ErrTooLarge):
		return -6
	case errors.Is(err, broker.ErrBufferSmall):
		return -7
	case errors.Is(err, broker.ErrUnavailable):
		return -8
	case errors.Is(err, broker.ErrClosed):
		return -9
	case errors.Is(err, broker.ErrLoginURL):
		return -10
	default:
		return -100
	}
}

func lookup(handle C.int64_t) *broker.Core {
	managerMu.Lock()
	defer managerMu.Unlock()
	return cores[uint64(handle)]
}

func writeJSON(out *C.char, capacity C.int32_t, value any) C.int32_t {
	data, err := json.Marshal(value)
	if err != nil {
		return setError(err)
	}
	if int(capacity) < len(data) {
		return setError(broker.ErrBufferSmall)
	}
	copy(unsafe.Slice((*byte)(unsafe.Pointer(out)), int(capacity)), data)
	return C.int32_t(len(data))
}

//export nanobot_network_abi_version
func nanobot_network_abi_version() C.int32_t {
	return 1
}

//export nanobot_network_start
func nanobot_network_start(configJSON *C.char) C.int64_t {
	var config broker.Config
	if err := json.Unmarshal([]byte(C.GoString(configJSON)), &config); err != nil {
		setError(broker.ErrInvalid)
		return 0
	}
	core, err := broker.NewCore(config)
	if err != nil {
		setError(err)
		return 0
	}
	if err := core.Start(); err != nil {
		setError(err)
		return 0
	}
	managerMu.Lock()
	nextHandle++
	handle := nextHandle
	cores[handle] = core
	managerMu.Unlock()
	return C.int64_t(handle)
}

//export nanobot_network_status
func nanobot_network_status(handle C.int64_t, out *C.char, capacity C.int32_t) C.int32_t {
	core := lookup(handle)
	if core == nil {
		return setError(broker.ErrNotFound)
	}
	return writeJSON(out, capacity, core.Status())
}

//export nanobot_network_login_url
func nanobot_network_login_url(handle C.int64_t, out *C.char, capacity C.int32_t) C.int32_t {
	core := lookup(handle)
	if core == nil {
		return setError(broker.ErrNotFound)
	}
	loginURL := core.LoginURL()
	if loginURL == "" {
		return setError(broker.ErrUnavailable)
	}
	return writeJSON(out, capacity, map[string]any{"login_url": loginURL})
}

//export nanobot_network_http
func nanobot_network_http(handle C.int64_t, requestJSON *C.char, outID *C.uint64_t) C.int32_t {
	core := lookup(handle)
	if core == nil {
		return setError(broker.ErrNotFound)
	}
	var request broker.HTTPRequest
	if err := json.Unmarshal([]byte(C.GoString(requestJSON)), &request); err != nil {
		return setError(broker.ErrInvalid)
	}
	id, err := core.HTTP(request)
	if err != nil {
		return setError(err)
	}
	*outID = C.uint64_t(id)
	return 0
}

//export nanobot_network_cancel
func nanobot_network_cancel(handle C.int64_t, requestID C.uint64_t) C.int32_t {
	core := lookup(handle)
	if core == nil {
		return setError(broker.ErrNotFound)
	}
	if err := core.Cancel(uint64(requestID)); err != nil {
		return setError(err)
	}
	return 0
}

//export nanobot_network_ws_open
func nanobot_network_ws_open(handle C.int64_t, requestJSON *C.char, outID *C.uint64_t) C.int32_t {
	core := lookup(handle)
	if core == nil {
		return setError(broker.ErrNotFound)
	}
	var request broker.WSRequest
	if err := json.Unmarshal([]byte(C.GoString(requestJSON)), &request); err != nil {
		return setError(broker.ErrInvalid)
	}
	id, err := core.WSOpen(request)
	if err != nil {
		return setError(err)
	}
	*outID = C.uint64_t(id)
	return 0
}

//export nanobot_network_ws_send
func nanobot_network_ws_send(handle C.int64_t, wsID C.uint64_t, kind C.int32_t, data *C.uint8_t, length C.int32_t) C.int32_t {
	core := lookup(handle)
	if core == nil {
		return setError(broker.ErrNotFound)
	}
	if length < 0 {
		return setError(broker.ErrInvalid)
	}
	var frame []byte
	if length > 0 {
		frame = C.GoBytes(unsafe.Pointer(data), length)
	}
	if err := core.WSSend(uint64(wsID), int(kind), frame); err != nil {
		return setError(err)
	}
	return 0
}

//export nanobot_network_ws_close
func nanobot_network_ws_close(handle C.int64_t, wsID C.uint64_t) C.int32_t {
	core := lookup(handle)
	if core == nil {
		return setError(broker.ErrNotFound)
	}
	if err := core.WSClose(uint64(wsID)); err != nil {
		return setError(err)
	}
	return 0
}

//export nanobot_network_ws_ping
func nanobot_network_ws_ping(handle C.int64_t, wsID C.uint64_t) C.int32_t {
	core := lookup(handle)
	if core == nil {
		return setError(broker.ErrNotFound)
	}
	if err := core.WSPing(uint64(wsID)); err != nil {
		return setError(err)
	}
	return 0
}

//export nanobot_network_poll_event
func nanobot_network_poll_event(handle C.int64_t, out *C.char, capacity C.int32_t) C.int32_t {
	core := lookup(handle)
	if core == nil {
		return setError(broker.ErrNotFound)
	}
	if capacity <= 0 {
		return setError(broker.ErrBufferSmall)
	}
	buffer := unsafe.Slice((*byte)(unsafe.Pointer(out)), int(capacity))
	n, err := core.PollEvent(buffer)
	if err != nil {
		return setError(err)
	}
	return C.int32_t(n)
}

//export nanobot_network_stop
func nanobot_network_stop(handle C.int64_t) C.int32_t {
	managerMu.Lock()
	core := cores[uint64(handle)]
	delete(cores, uint64(handle))
	managerMu.Unlock()
	if core == nil {
		return setError(broker.ErrNotFound)
	}
	if err := core.Stop(); err != nil {
		return setError(err)
	}
	return 0
}

//export nanobot_network_last_error
func nanobot_network_last_error() *C.char {
	managerMu.Lock()
	defer managerMu.Unlock()
	if lastErrC == nil {
		lastErrC = C.CString("")
	}
	return lastErrC
}
