package mnist

import (
	"errors"
)

type ImageReader struct {
	imagesReader *Reader
	labelsReader *Reader
	ImageCount   int32
	RowCount     int32
	ColCount     int32
}

type Image struct {
	Label  int
	Pixels []uint8
}

var EUnmatchedTrainingSets = errors.New("unmatched amount of training sets")

func NewImageReader(imagesPath string, labelsPath string) (*ImageReader, error) {
	imagesReader, err := NewReader(imagesPath, 0x803, 3)
	if err != nil {
		return nil, err
	}
	labelsReader, err := NewReader(labelsPath, 0x801, 1)
	if err != nil {
		return nil, err
	}

	r := &ImageReader{
		imagesReader: imagesReader,
		labelsReader: labelsReader,
	}

	// Metadata
	r.ImageCount = imagesReader.Meta[0]
	if labelsReader.Meta[0] != r.ImageCount {
		return nil, EUnmatchedTrainingSets
	}
	r.RowCount = imagesReader.Meta[1]
	r.ColCount = imagesReader.Meta[2]

	return r, nil
}

func (r *ImageReader) PixelCount() int32 {
	return r.ColCount * r.RowCount
}

func (r *ImageReader) Next() (image *Image, err error) {
	buf := make([]byte, 1)
	_, err = r.labelsReader.Read(buf)
	if err != nil {
		return nil, err
	}
	label := int(buf[0])

	pixels := make([]byte, r.PixelCount())
	_, err = r.imagesReader.Read(pixels)
	if err != nil {
		return nil, err
	}

	return &Image{
		label,
		pixels,
	}, nil
}

func (r *ImageReader) Close() {
	r.imagesReader.Close()
	r.labelsReader.Close()
}
