package broker

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"

	"tailscale.com/ipn"
)

const (
	stateMagic   = "NBS1"
	stateVersion = 1
	stateAAD     = "nanobot-state-v1"
)

// EncryptedFileStore implements ipn.StateStore with an AES-256-GCM encrypted
// map file. The key is generated and wrapped by Android Keystore in Kotlin and
// passed to the native transport only at start.
type EncryptedFileStore struct {
	dir  string
	path string
	key  []byte
	mu   sync.Mutex
}

func NewEncryptedFileStore(dir string, key []byte) (*EncryptedFileStore, error) {
	if len(key) != 32 {
		return nil, fmt.Errorf("%w: state key must be 32 bytes", ErrInvalid)
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	if err := os.Chmod(dir, 0o700); err != nil {
		return nil, err
	}
	return &EncryptedFileStore{
		dir:  dir,
		path: filepath.Join(dir, "state.enc"),
		key:  append([]byte(nil), key...),
	}, nil
}

func (s *EncryptedFileStore) ReadState(id ipn.StateKey) ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	values, err := s.read()
	if err != nil {
		return nil, err
	}
	value, ok := values[string(id)]
	if !ok {
		return nil, ipn.ErrStateNotExist
	}
	return append([]byte(nil), value...), nil
}

func (s *EncryptedFileStore) WriteState(id ipn.StateKey, value []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	values, err := s.read()
	if err != nil {
		return err
	}
	if value == nil {
		delete(values, string(id))
	} else {
		values[string(id)] = append([]byte(nil), value...)
	}
	return s.write(values)
}

func (s *EncryptedFileStore) read() (map[string][]byte, error) {
	data, err := os.ReadFile(s.path)
	if errors.Is(err, os.ErrNotExist) {
		return map[string][]byte{}, nil
	}
	if err != nil {
		return nil, err
	}
	if len(data) < 5 || string(data[:4]) != stateMagic {
		return nil, fmt.Errorf("%w: state file magic is invalid", ErrInvalid)
	}
	if data[4] != stateVersion {
		return nil, fmt.Errorf("%w: unknown state version %d", ErrInvalid, data[4])
	}
	if len(data) < 5+12+16 {
		return nil, fmt.Errorf("%w: state file is truncated", ErrInvalid)
	}
	nonce := data[5:17]
	ciphertext := data[17:]
	aead, err := s.aead()
	if err != nil {
		return nil, err
	}
	plaintext, err := aead.Open(nil, nonce, ciphertext, []byte(stateAAD))
	if err != nil {
		return nil, fmt.Errorf("%w: state file cannot be decrypted", ErrInvalid)
	}
	values := map[string][]byte{}
	if err := json.Unmarshal(plaintext, &values); err != nil {
		return nil, fmt.Errorf("%w: state file is not valid JSON", ErrInvalid)
	}
	return values, nil
}

func (s *EncryptedFileStore) write(values map[string][]byte) error {
	if len(values) == 0 {
		if err := os.Remove(s.path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		return s.syncDir()
	}
	plaintext, err := json.Marshal(values)
	if err != nil {
		return err
	}
	aead, err := s.aead()
	if err != nil {
		return err
	}
	nonce := make([]byte, aead.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return err
	}
	ciphertext := aead.Seal(nil, nonce, plaintext, []byte(stateAAD))
	payload := make([]byte, 0, 5+len(nonce)+len(ciphertext))
	payload = append(payload, stateMagic...)
	payload = append(payload, stateVersion)
	payload = append(payload, nonce...)
	payload = append(payload, ciphertext...)

	temp, err := os.CreateTemp(s.dir, ".state-*.tmp")
	if err != nil {
		return err
	}
	tempName := temp.Name()
	defer os.Remove(tempName)
	if err := temp.Chmod(0o600); err != nil {
		temp.Close()
		return err
	}
	if _, err := temp.Write(payload); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Sync(); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tempName, s.path); err != nil {
		return err
	}
	return s.syncDir()
}

func (s *EncryptedFileStore) aead() (cipher.AEAD, error) {
	block, err := aes.NewCipher(s.key)
	if err != nil {
		return nil, err
	}
	return cipher.NewGCM(block)
}

func (s *EncryptedFileStore) syncDir() error {
	handle, err := os.Open(s.dir)
	if err != nil {
		return err
	}
	defer handle.Close()
	return handle.Sync()
}
