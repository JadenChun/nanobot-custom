package broker

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"tailscale.com/ipn"
)

func TestEncryptedStateStoreRoundTripAndFailureModes(t *testing.T) {
	dir := t.TempDir()
	key := make([]byte, 32)
	for index := range key {
		key[index] = byte(index)
	}
	store, err := NewEncryptedFileStore(dir, key)
	if err != nil {
		t.Fatalf("NewEncryptedFileStore: %v", err)
	}
	if _, err := store.ReadState(ipn.StateKey("profile")); !errors.Is(err, ipn.ErrStateNotExist) {
		t.Fatalf("empty read: %v", err)
	}
	if err := store.WriteState(ipn.StateKey("profile"), []byte("NANOBOT_SECRET")); err != nil {
		t.Fatalf("write: %v", err)
	}
	value, err := store.ReadState(ipn.StateKey("profile"))
	if err != nil || string(value) != "NANOBOT_SECRET" {
		t.Fatalf("read: %q %v", value, err)
	}
	info, err := os.Stat(filepath.Join(dir, "state.enc"))
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("state mode: %o", info.Mode().Perm())
	}
	raw, err := os.ReadFile(filepath.Join(dir, "state.enc"))
	if err != nil {
		t.Fatalf("read raw: %v", err)
	}
	for index := 0; index+len("NANOBOT_SECRET") <= len(raw); index++ {
		if string(raw[index:index+len("NANOBOT_SECRET")]) == "NANOBOT_SECRET" {
			t.Fatal("plaintext secret appeared in state file")
		}
	}

	reopened, err := NewEncryptedFileStore(dir, key)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if value, err := reopened.ReadState(ipn.StateKey("profile")); err != nil || string(value) != "NANOBOT_SECRET" {
		t.Fatalf("reopen read: %q %v", value, err)
	}

	wrongKey := make([]byte, 32)
	wrongKey[0] = 0xff
	wrong, err := NewEncryptedFileStore(dir, wrongKey)
	if err != nil {
		t.Fatalf("wrong key store: %v", err)
	}
	if _, err := wrong.ReadState(ipn.StateKey("profile")); !errors.Is(err, ErrInvalid) {
		t.Fatalf("wrong key error: %v", err)
	}

	if err := store.WriteState(ipn.StateKey("profile"), nil); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if _, err := store.ReadState(ipn.StateKey("profile")); !errors.Is(err, ipn.ErrStateNotExist) {
		t.Fatalf("delete read: %v", err)
	}
	if err := store.WriteState(ipn.StateKey("profile"), []byte("NANOBOT_SECRET")); err != nil {
		t.Fatalf("rewrite: %v", err)
	}
	raw, err = os.ReadFile(filepath.Join(dir, "state.enc"))
	if err != nil {
		t.Fatalf("read raw again: %v", err)
	}
	raw[4] = 99
	if err := os.WriteFile(filepath.Join(dir, "state.enc"), raw, 0o600); err != nil {
		t.Fatalf("corrupt write: %v", err)
	}
	if _, err := reopened.ReadState(ipn.StateKey("profile")); !errors.Is(err, ErrInvalid) {
		t.Fatalf("unknown version error: %v", err)
	}
}

func TestStartRejectsMissingOrShortStateKey(t *testing.T) {
	cfg := Config{
		Origin:         "https://fixture.example:8443",
		InstallationID: "fixture-installation",
		Hostname:       "fixture-host",
		StateDir:       t.TempDir(),
	}
	core, err := NewCore(cfg)
	if err != nil {
		t.Fatalf("NewCore: %v", err)
	}
	if err := core.Start(); !errors.Is(err, ErrInvalid) {
		t.Fatalf("missing key start: %v", err)
	}
	cfg.StateKeyBase64 = "c2hvcnQ="
	short, err := NewCore(cfg)
	if err != nil {
		t.Fatalf("NewCore short: %v", err)
	}
	if err := short.Start(); !errors.Is(err, ErrInvalid) {
		t.Fatalf("short key start: %v", err)
	}
}
