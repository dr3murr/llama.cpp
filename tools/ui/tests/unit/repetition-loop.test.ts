import { findRepeatedSuffix } from '$lib/utils/repetition-loop';
import { describe, expect, it } from 'vitest';

describe('findRepeatedSuffix', () => {
	const cycle =
		"I'm realizing the compiler might have replaced the instruction with a load. " +
		'Let me check the actual bytes in the binary rather than decoding it by hand. ';

	it('detects a long reasoning cycle after three repetitions', () => {
		const result = findRepeatedSuffix(`Useful prefix. ${cycle.repeat(3)}`);

		expect(result).toEqual({
			periodChars: cycle.length,
			repeatedChars: cycle.length * 3,
			repetitions: 3
		});
	});

	it('detects a cycle when the streaming suffix ends partway through its natural phrase', () => {
		const rotated = cycle.slice(37) + cycle.slice(0, 37);
		const result = findRepeatedSuffix(`Useful prefix. ${rotated.repeat(3)}`);

		expect(result?.periodChars).toBe(cycle.length);
	});

	it('ignores short repetitions and ordinary duplicated paragraphs', () => {
		expect(findRepeatedSuffix('no '.repeat(100))).toBeNull();
		expect(findRepeatedSuffix(`prefix ${cycle}${cycle}`)).toBeNull();
	});

	it('bounds work to the configured suffix window', () => {
		const oldCycle = cycle.repeat(3);
		const nonRepeatingTail = Array.from({ length: 500 }, (_, index) => `tail-${index}`).join('|');

		expect(
			findRepeatedSuffix(`${oldCycle}${nonRepeatingTail}`, { maxWindowChars: 2048 })
		).toBeNull();
	});
});
