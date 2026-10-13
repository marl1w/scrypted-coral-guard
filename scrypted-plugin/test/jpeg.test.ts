import test from 'node:test';
import assert from 'node:assert/strict';
import { JpegSplitter, qualityToQscale } from '../src/jpeg';

const jpg = (n: number) => Buffer.from([0xff, 0xd8, n, n, 0xff, 0x00, 0xff, 0xd9]);
const split = (chunks: Buffer[]) => { const s = new JpegSplitter(); const out: Buffer[] = []; for (const c of chunks) s.push(c, f => out.push(f)); return out; };

test('splits concatenated JPEGs, also across chunk boundaries', () => {
    const all = Buffer.concat([jpg(1), jpg(2), jpg(3)]);
    assert.deepEqual(split([all.subarray(0, 4), all.subarray(4, 11), all.subarray(11)]), [jpg(1), jpg(2), jpg(3)]);
});

test('every split point of a two-frame stream gives the same frames (marker halves)', () => {
    const all = Buffer.concat([jpg(1), jpg(2)]);
    for (let i = 1; i < all.length; i++)
        assert.deepEqual(split([all.subarray(0, i), all.subarray(i)]), [jpg(1), jpg(2)], `split at ${i}`);
});

test('byte-by-byte feed works', () => {
    const all = Buffer.concat([jpg(1), jpg(2), jpg(3)]);
    assert.deepEqual(split([...all].map(b => Buffer.from([b]))), [jpg(1), jpg(2), jpg(3)]);
});

test('ignores garbage before SOI and holds a partial frame', () => {
    const s = new JpegSplitter();
    const out: Buffer[] = [];
    s.push(Buffer.from([0, 1, 2, 0xff, 0xd8, 9]), f => out.push(f));
    assert.equal(out.length, 0);
    s.push(Buffer.from([0xff, 0xd9]), f => out.push(f));
    assert.deepEqual(out, [Buffer.from([0xff, 0xd8, 9, 0xff, 0xd9])]);
});

test('handed-out frames stay valid after the accumulator is reused; large frames grow it', () => {
    const big = Buffer.concat([Buffer.from([0xff, 0xd8]), Buffer.alloc(600000, 7), Buffer.from([0xff, 0xd9])]);
    const out = split([big, jpg(1), big]);
    assert.equal(out.length, 3);
    assert.ok(out[0].equals(big) && out[2].equals(big) && out[1].equals(jpg(1)));
});

test('quality to qscale: 85 -> 6, bounds', () => {
    assert.equal(qualityToQscale(85), 6);
    assert.equal(qualityToQscale(100), 2);
    assert.equal(qualityToQscale(1), 31);
});
