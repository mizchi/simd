// @simdimage tests — run on all four targets; assert the SIMD (wasm) and
// scalar (wasm-gc / native / js) paths produce identical results.

///|
fn bytes_of(arr : FixedArray[Byte]) -> Bytes {
  Bytes::from_array(arr)
}

///|
test "rgb_to_rgba" {
  let rgb = bytes_of([
    b'\xFF', b'\x00', b'\x00', // red
     b'\x00', b'\xFF', b'\x00', // green
     b'\x00', b'\x00', b'\xFF', // blue
  ])
  let rgba = FixedArray::make(12, b'\x00')
  rgb_to_rgba(rgb, b'\xFF', rgba)
  let r = rgba
  assert_true(
    (
      r[0].to_int(),
      r[1].to_int(),
      r[2].to_int(),
      r[3].to_int(),
      r[4].to_int(),
      r[5].to_int(),
      r[6].to_int(),
      r[7].to_int(),
      r[8].to_int(),
      r[9].to_int(),
      r[10].to_int(),
      r[11].to_int(),
    ) ==
    (255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255),
  )
}

///|
test "rgb_to_rgba SIMD body + tail" {
  for npix in [0, 1, 3, 4, 5, 7, 8, 9, 16, 17] {
    let rgb = FixedArray::make(npix * 3, b'\x00')
    for i in 0..<npix {
      rgb[i * 3] = ((i * 3 + 1) % 256).to_byte()
      rgb[i * 3 + 1] = ((i * 7 + 2) % 256).to_byte()
      rgb[i * 3 + 2] = ((i * 13 + 3) % 256).to_byte()
    }
    let src = bytes_of(rgb)
    let rgba = FixedArray::make(npix * 4, b'\x00')
    rgb_to_rgba(src, b'\xAB', rgba)
    for i in 0..<npix {
      assert_true(rgba[i * 4] == rgb[i * 3])
      assert_true(rgba[i * 4 + 1] == rgb[i * 3 + 1])
      assert_true(rgba[i * 4 + 2] == rgb[i * 3 + 2])
      assert_true(rgba[i * 4 + 3].to_int() == 0xAB)
    }
  }
}

///|
test "rgba_to_grayscale" {
  let rgba = bytes_of([
    b'\xFF', b'\xFF', b'\xFF', b'\xFF', // white -> 255
     b'\x00', b'\x00', b'\x00', b'\xFF', // black -> 0
     b'\xFF', b'\x00', b'\x00', b'\xFF', // red -> 76
  ])
  let gray = FixedArray::make(3, b'\x00')
  rgba_to_grayscale(rgba, gray)
  assert_true(
    (gray[0].to_int(), gray[1].to_int(), gray[2].to_int()) == (255, 0, 76),
  )
}

///|
test "channel_extract / channel_merge round-trip" {
  let original = bytes_of([
    b'\x01', b'\x02', b'\x03', b'\x04', b'\x05', b'\x06', b'\x07', b'\x08',
  ])
  let r_ch = FixedArray::make(2, b'\x00')
  let g_ch = FixedArray::make(2, b'\x00')
  let b_ch = FixedArray::make(2, b'\x00')
  let a_ch = FixedArray::make(2, b'\x00')
  channel_extract(original, 0, r_ch)
  channel_extract(original, 1, g_ch)
  channel_extract(original, 2, b_ch)
  channel_extract(original, 3, a_ch)
  let rebuilt = FixedArray::make(8, b'\x00')
  channel_merge(
    bytes_of(r_ch),
    bytes_of(g_ch),
    bytes_of(b_ch),
    bytes_of(a_ch),
    rebuilt,
  )
  for i in 0..<8 {
    assert_true(rebuilt[i] == original[i])
  }
}

///|
test "lerp at t=0, t=128, t=256" {
  let a = bytes_of([b'\x00', b'\x10', b'\xFF'])
  let b = bytes_of([b'\xFF', b'\x80', b'\x00'])
  let out = FixedArray::make(3, b'\x00')
  lerp(a, b, 0, out)
  assert_true(
    (out[0].to_int(), out[1].to_int(), out[2].to_int()) == (0, 16, 255),
  )
  lerp(a, b, 256, out)
  assert_true(
    (out[0].to_int(), out[1].to_int(), out[2].to_int()) == (255, 128, 0),
  )
  lerp(a, b, 128, out)
  assert_true(
    (out[0].to_int(), out[1].to_int(), out[2].to_int()) == (127, 72, 127),
  )
}

///|
test "histogram counts byte occurrences" {
  let data = bytes_of([b'\x00', b'\x05', b'\x05', b'\xFF', b'\x05'])
  let bins = FixedArray::make(256, 0)
  histogram(data, bins)
  assert_true(bins[0] == 1)
  assert_true(bins[5] == 3)
  assert_true(bins[255] == 1)
}

///|
test "alpha_blend_solid: src over transparent dst" {
  let dst = FixedArray::make(8, b'\x00')
  // pixel 0 transparent, pixel 1 opaque white
  for c in 4..<8 {
    dst[c] = b'\xFF'
  }
  alpha_blend_solid(dst, b'\xFF', b'\x00', b'\x00', b'\xFF') // opaque red
  assert_true(
    (
      dst[0].to_int(),
      dst[1].to_int(),
      dst[2].to_int(),
      dst[3].to_int(),
      dst[4].to_int(),
      dst[5].to_int(),
      dst[6].to_int(),
      dst[7].to_int(),
    ) ==
    (255, 0, 0, 255, 255, 0, 0, 255),
  )
}

///|
test "alpha_blend_solid: half-alpha src over white" {
  let dst = FixedArray::make(4, b'\xFF')
  alpha_blend_solid(dst, b'\x80', b'\x00', b'\x00', b'\x80')
  assert_true(
    (dst[0].to_int(), dst[1].to_int(), dst[2].to_int(), dst[3].to_int()) ==
    (255, 127, 127, 255),
  )
}

///|
test "channel_extract SIMD body + tail (all channels)" {
  for npix in [0, 1, 3, 4, 7, 8, 13, 16, 17] {
    let raw = FixedArray::make(npix * 4, b'\x00')
    for i in 0..<npix {
      raw[i * 4] = (i % 256).to_byte()
      raw[i * 4 + 1] = ((i * 2 + 1) % 256).to_byte()
      raw[i * 4 + 2] = ((i * 3 + 2) % 256).to_byte()
      raw[i * 4 + 3] = ((i * 5 + 7) % 256).to_byte()
    }
    let rgba = bytes_of(raw)
    for ch in 0..<4 {
      let out = FixedArray::make(npix, b'\x00')
      channel_extract(rgba, ch, out)
      for i in 0..<npix {
        assert_true(out[i] == raw[i * 4 + ch])
      }
    }
  }
}

///|
test "channel_merge SIMD body + tail" {
  for npix in [0, 1, 3, 4, 7, 8, 13, 16, 17] {
    let r = FixedArray::make(npix, b'\x00')
    let g = FixedArray::make(npix, b'\x00')
    let b = FixedArray::make(npix, b'\x00')
    let a = FixedArray::make(npix, b'\x00')
    for i in 0..<npix {
      r[i] = (i % 256).to_byte()
      g[i] = ((i * 2 + 1) % 256).to_byte()
      b[i] = ((i * 3 + 2) % 256).to_byte()
      a[i] = ((i * 5 + 7) % 256).to_byte()
    }
    let out = FixedArray::make(npix * 4, b'\x00')
    channel_merge(bytes_of(r), bytes_of(g), bytes_of(b), bytes_of(a), out)
    for i in 0..<npix {
      assert_true(out[i * 4] == r[i])
      assert_true(out[i * 4 + 1] == g[i])
      assert_true(out[i * 4 + 2] == b[i])
      assert_true(out[i * 4 + 3] == a[i])
    }
  }
}

///|
test "lerp SIMD body + tail + edges" {
  for len in [0, 1, 15, 16, 17, 33, 100] {
    let af = FixedArray::make(len, b'\x00')
    let bf = FixedArray::make(len, b'\x00')
    for i in 0..<len {
      af[i] = (i * 3 % 256).to_byte()
      bf[i] = ((255 - i * 7 % 256) % 256).to_byte()
    }
    let a = bytes_of(af)
    let b = bytes_of(bf)
    for t in [0, 1, 64, 128, 200, 255, 256, -5, 300] {
      let out = FixedArray::make(len, b'\x00')
      lerp(a, b, t, out)
      let tc = if t < 0 { 0 } else if t > 256 { 256 } else { t }
      let inv = 256 - tc
      for i in 0..<len {
        let want = (af[i].to_int() * inv + bf[i].to_int() * tc) >> 8
        assert_true(out[i].to_int() == want)
      }
    }
  }
}

///|
test "rgba_to_grayscale SIMD body + tail" {
  for npix in [0, 1, 3, 4, 7, 8, 13, 16, 17] {
    let raw = FixedArray::make(npix * 4, b'\x00')
    for i in 0..<npix {
      raw[i * 4] = (i * 11 % 256).to_byte()
      raw[i * 4 + 1] = (i * 29 % 256).to_byte()
      raw[i * 4 + 2] = (i * 47 % 256).to_byte()
      raw[i * 4 + 3] = b'\xFF'
    }
    let rgba = bytes_of(raw)
    let out = FixedArray::make(npix, b'\x00')
    rgba_to_grayscale(rgba, out)
    for i in 0..<npix {
      let r = raw[i * 4].to_int()
      let g = raw[i * 4 + 1].to_int()
      let b = raw[i * 4 + 2].to_int()
      assert_true(out[i].to_int() == (77 * r + 150 * g + 29 * b) >> 8)
    }
  }
}

///|
test "alpha_blend_solid SIMD body + tail vs reference" {
  fn blend(d : Int, inv : Int, sc : Int) -> Int {
    let t = d * inv + 128
    let r = (t + (t >> 8)) >> 8
    let o = sc + r
    if o > 255 {
      255
    } else {
      o
    }
  }

  for npix in [0, 1, 3, 4, 7, 8, 13, 16, 17] {
    let dst = FixedArray::make(npix * 4, b'\x00')
    for i in 0..<npix {
      dst[i * 4] = (i * 17 % 256).to_byte()
      dst[i * 4 + 1] = (i * 53 % 256).to_byte()
      dst[i * 4 + 2] = (i * 97 % 256).to_byte()
      dst[i * 4 + 3] = (i * 211 % 256).to_byte()
    }
    let src = [200, 100, 50, 128]
    let inv = 255 - src[3]
    let want = FixedArray::make(npix * 4, b'\x00')
    for i in 0..<npix {
      for c in 0..<4 {
        want[i * 4 + c] = blend(dst[i * 4 + c].to_int(), inv, src[c]).to_byte()
      }
    }
    alpha_blend_solid(
      dst,
      src[0].to_byte(),
      src[1].to_byte(),
      src[2].to_byte(),
      src[3].to_byte(),
    )
    for i in 0..<(npix * 4) {
      assert_true(dst[i] == want[i])
    }
  }
}
