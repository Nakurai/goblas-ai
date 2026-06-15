package model

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"time"
)

// FormatVersion is the version of the .gobl container layout. It is written into
// every file so that future readers can detect and reject formats they do not
// understand.
const FormatVersion = 1

// magic identifies a goblas-ai model file.
var magic = [8]byte{'G', 'O', 'B', 'L', 'A', 'I', 0, 1}

// Metadata is the self-describing header stored in every model file. It can be
// read without the algorithm's package being imported, which is what powers
// `goblasai inspect`.
type Metadata struct {
	Algorithm     string         `json:"algorithm"`
	FormatVersion int            `json:"format_version"`
	CreatedAt     string         `json:"created_at"`
	Params        map[string]any `json:"params"`
}

// Save writes a trained model to w in the .gobl container format: magic bytes, a
// format version, a JSON metadata header, and the algorithm's own serialized
// weights.
func Save(w io.Writer, m Persistable) error {
	weights, err := m.MarshalWeights()
	if err != nil {
		return fmt.Errorf("model: marshal weights: %w", err)
	}
	meta := Metadata{
		Algorithm:     m.Algorithm(),
		FormatVersion: FormatVersion,
		CreatedAt:     time.Now().UTC().Format(time.RFC3339),
		Params:        m.Params(),
	}
	metaBytes, err := json.Marshal(meta)
	if err != nil {
		return fmt.Errorf("model: marshal metadata: %w", err)
	}

	if _, err := w.Write(magic[:]); err != nil {
		return err
	}
	if err := binary.Write(w, binary.BigEndian, uint32(FormatVersion)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.BigEndian, uint32(len(metaBytes))); err != nil {
		return err
	}
	if _, err := w.Write(metaBytes); err != nil {
		return err
	}
	if err := binary.Write(w, binary.BigEndian, uint64(len(weights))); err != nil {
		return err
	}
	if _, err := w.Write(weights); err != nil {
		return err
	}
	return nil
}

// SaveFile writes a trained model to the file at path.
func SaveFile(path string, m Persistable) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("model: create file: %w", err)
	}
	if err := Save(f, m); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// Load reads a model from r and returns it as a Predictor, ready for inference.
// The algorithm's package must have been imported so that it is registered (see
// Register).
func Load(r io.Reader) (Predictor, error) {
	instance, err := LoadModel(r)
	if err != nil {
		return nil, err
	}
	predictor, ok := instance.(Predictor)
	if !ok {
		return nil, fmt.Errorf("model: algorithm %q does not support row-by-row prediction", instance.Algorithm())
	}
	return predictor, nil
}

// LoadFile reads a model from the file at path and returns it as a Predictor.
func LoadFile(path string) (Predictor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("model: open file: %w", err)
	}
	defer f.Close()
	return Load(f)
}

// LoadModel reads a model from r and returns it as a Persistable, without
// requiring it to support row-by-row prediction. Algorithms that are not the
// simple Predictor shape — such as the stateful, sequence-based NG-RC model —
// are loaded this way and then used through their own methods. The algorithm's
// package must have been imported so that it is registered (see Register).
func LoadModel(r io.Reader) (Persistable, error) {
	meta, weights, err := readContainer(r)
	if err != nil {
		return nil, err
	}
	factory, err := lookup(meta.Algorithm)
	if err != nil {
		return nil, err
	}
	instance := factory()
	if err := instance.UnmarshalWeights(weights); err != nil {
		return nil, fmt.Errorf("model: unmarshal weights: %w", err)
	}
	return instance, nil
}

// LoadModelFile reads a model from the file at path as a Persistable.
func LoadModelFile(path string) (Persistable, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("model: open file: %w", err)
	}
	defer f.Close()
	return LoadModel(f)
}

// ReadMetadata reads only the descriptive header of a model file, without
// needing the algorithm's package to be imported.
func ReadMetadata(r io.Reader) (Metadata, error) {
	meta, _, err := readContainer(r)
	return meta, err
}

// ReadMetadataFile reads the header of the model file at path.
func ReadMetadataFile(path string) (Metadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return Metadata{}, fmt.Errorf("model: open file: %w", err)
	}
	defer f.Close()
	return ReadMetadata(f)
}

// readContainer parses the container layout and returns the metadata and the raw
// weight bytes.
func readContainer(r io.Reader) (Metadata, []byte, error) {
	var gotMagic [8]byte
	if _, err := io.ReadFull(r, gotMagic[:]); err != nil {
		return Metadata{}, nil, fmt.Errorf("model: read magic: %w", err)
	}
	if !bytes.Equal(gotMagic[:], magic[:]) {
		return Metadata{}, nil, fmt.Errorf("model: not a goblas-ai model file")
	}

	var version uint32
	if err := binary.Read(r, binary.BigEndian, &version); err != nil {
		return Metadata{}, nil, err
	}
	if version > FormatVersion {
		return Metadata{}, nil, fmt.Errorf("model: file format version %d is newer than supported version %d", version, FormatVersion)
	}

	var metaLen uint32
	if err := binary.Read(r, binary.BigEndian, &metaLen); err != nil {
		return Metadata{}, nil, err
	}
	metaBytes := make([]byte, metaLen)
	if _, err := io.ReadFull(r, metaBytes); err != nil {
		return Metadata{}, nil, fmt.Errorf("model: read metadata: %w", err)
	}
	var meta Metadata
	if err := json.Unmarshal(metaBytes, &meta); err != nil {
		return Metadata{}, nil, fmt.Errorf("model: parse metadata: %w", err)
	}

	var weightsLen uint64
	if err := binary.Read(r, binary.BigEndian, &weightsLen); err != nil {
		return Metadata{}, nil, err
	}
	weights := make([]byte, weightsLen)
	if _, err := io.ReadFull(r, weights); err != nil {
		return Metadata{}, nil, fmt.Errorf("model: read weights: %w", err)
	}
	return meta, weights, nil
}
