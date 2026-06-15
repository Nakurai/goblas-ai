package onnx

import (
	"encoding/binary"
	"math"
)

// A tiny, dependency-free Protocol Buffers writer — just enough to emit the
// handful of ONNX message types a linear model needs. ONNX files are encoded
// with protobuf, and the op set for linear regression is small enough that
// hand-writing the bytes is far lighter than pulling in a protobuf toolchain.

// wire types
const (
	wireVarint  = 0
	wireFixed32 = 5
	wireLen     = 2
)

type pbuf struct{ b []byte }

func (p *pbuf) tag(field, wire int) {
	p.uvarint(uint64(field)<<3 | uint64(wire))
}

func (p *pbuf) uvarint(v uint64) {
	for v >= 0x80 {
		p.b = append(p.b, byte(v)|0x80)
		v >>= 7
	}
	p.b = append(p.b, byte(v))
}

// int64Field writes a varint-encoded int64 field.
func (p *pbuf) int64Field(field int, v int64) {
	p.tag(field, wireVarint)
	p.uvarint(uint64(v))
}

// stringField writes a length-delimited string field.
func (p *pbuf) stringField(field int, s string) {
	p.bytesField(field, []byte(s))
}

// bytesField writes a length-delimited bytes field.
func (p *pbuf) bytesField(field int, data []byte) {
	p.tag(field, wireLen)
	p.uvarint(uint64(len(data)))
	p.b = append(p.b, data...)
}

// messageField writes an embedded message (length-delimited).
func (p *pbuf) messageField(field int, sub []byte) {
	p.bytesField(field, sub)
}

// floatField writes a 32-bit float field.
func (p *pbuf) floatField(field int, f float32) {
	p.tag(field, wireFixed32)
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], math.Float32bits(f))
	p.b = append(p.b, buf[:]...)
}

// packedInt64 encodes a repeated int64 field using the packed representation.
func (p *pbuf) packedInt64(field int, vals []int64) {
	var inner pbuf
	for _, v := range vals {
		inner.uvarint(uint64(v))
	}
	p.bytesField(field, inner.b)
}

// packedFloats encodes a repeated float field using the packed representation.
func (p *pbuf) packedFloats(field int, vals []float32) {
	buf := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	p.bytesField(field, buf)
}
