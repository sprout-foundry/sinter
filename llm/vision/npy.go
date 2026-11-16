package vision

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// LoadNpyFloat32 reads a C-order little-endian float32 .npy file — the
// format numpy.save writes — and returns its flat data and shape.
//
// It exists so tests can compare against reference tensors dumped from
// Python without taking a numpy dependency or shipping a binary format of
// our own.
func LoadNpyFloat32(path string) ([]float32, []int, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	if len(raw) < 10 || string(raw[1:6]) != "NUMPY" {
		return nil, nil, fmt.Errorf("%s: not a .npy file", path)
	}
	var headerLen, headerStart int
	if raw[6] == 1 {
		headerLen = int(binary.LittleEndian.Uint16(raw[8:10]))
		headerStart = 10
	} else {
		headerLen = int(binary.LittleEndian.Uint32(raw[8:12]))
		headerStart = 12
	}
	if headerStart+headerLen > len(raw) {
		return nil, nil, fmt.Errorf("%s: truncated header", path)
	}
	header := string(raw[headerStart : headerStart+headerLen])
	payload := raw[headerStart+headerLen:]

	descr, err := npyField(header, "'descr':")
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", path, err)
	}
	if descr != "<f4" {
		return nil, nil, fmt.Errorf("%s: unsupported dtype %q, want <f4", path, descr)
	}

	shape, err := npyShape(header)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", path, err)
	}
	total := 1
	for _, d := range shape {
		total *= d
	}
	if len(payload) != total*4 {
		return nil, nil, fmt.Errorf("%s: payload %d bytes, want %d for shape %v", path, len(payload), total*4, shape)
	}
	out := make([]float32, total)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[i*4:]))
	}
	return out, shape, nil
}

// npyField returns the quoted string value following key in the header dict.
func npyField(header, key string) (string, error) {
	i := indexOf(header, key)
	if i < 0 {
		return "", fmt.Errorf("no %s in header", key)
	}
	rest := header[i+len(key):]
	q1 := indexByte(rest, '\'')
	if q1 < 0 {
		return "", fmt.Errorf("unterminated value after %s", key)
	}
	q2 := indexByte(rest[q1+1:], '\'')
	if q2 < 0 {
		return "", fmt.Errorf("unterminated value after %s", key)
	}
	return rest[q1+1 : q1+1+q2], nil
}

// npyShape parses the tuple after "'shape':".
func npyShape(header string) ([]int, error) {
	i := indexOf(header, "'shape':")
	if i < 0 {
		return nil, fmt.Errorf("no shape in header")
	}
	rest := header[i+len("'shape':"):]
	open := indexByte(rest, '(')
	close := indexByte(rest, ')')
	if open < 0 || close < 0 || close < open {
		return nil, fmt.Errorf("malformed shape in header")
	}
	var shape []int
	n, seen := 0, false
	for _, c := range rest[open+1 : close] {
		switch {
		case c >= '0' && c <= '9':
			n = n*10 + int(c-'0')
			seen = true
		case c == ',':
			if seen {
				shape = append(shape, n)
			}
			n, seen = 0, false
		}
	}
	if seen {
		shape = append(shape, n)
	}
	return shape, nil
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

func indexByte(s string, c byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			return i
		}
	}
	return -1
}
