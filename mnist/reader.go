package mnist

import (
	"encoding/binary"
	"errors"
	"os"
)

type Reader struct {
	Path     string
	file     *os.File
	MagicNum int32
	Meta     []int32
}

var EInvalidMagicNum = errors.New("magic number check failed")

func NewReader(path string, magicNum int32, metaCount int) (*Reader, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	r := &Reader{
		Path: path,
		file: file,
	}

	// Magic num
	buf := make([]byte, 4)
	_, err = r.file.Read(buf)
	if err != nil {
		return nil, err
	}
	r.MagicNum = int32(binary.BigEndian.Uint32(buf))
	if r.MagicNum != magicNum {
		return nil, EInvalidMagicNum
	}

	// Metadata
	buf = make([]byte, 4*metaCount)
	_, err = r.file.Read(buf)
	if err != nil {
		return nil, err
	}
	r.Meta = make([]int32, metaCount)
	for i := 0; i < metaCount; i++ {
		bytes := buf[i*4 : i*4+4]
		r.Meta[i] = int32(binary.BigEndian.Uint32(bytes))
	}

	return r, nil
}

func (r *Reader) Read(b []byte) (n int, err error) {
	return r.file.Read(b)
}

func (r *Reader) Close() {
	r.file.Close()
}
