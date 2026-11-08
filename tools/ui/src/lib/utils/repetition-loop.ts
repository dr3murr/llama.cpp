export interface RepeatedSuffix {
	periodChars: number;
	repeatedChars: number;
	repetitions: number;
}

export interface RepeatedSuffixOptions {
	maxPeriodChars?: number;
	maxWindowChars?: number;
	minPeriodChars?: number;
	repetitions?: number;
}

/**
 * Detect a long, exact cycle at the end of generated text.
 *
 * This intentionally operates only on the bounded suffix and requires several
 * complete repetitions. It is a streaming safety valve for model degeneration,
 * not a general duplicate-text detector: short repeated words, markdown rules,
 * source snippets and prompt/tool-result content are outside its scope.
 */
export function findRepeatedSuffix(
	text: string,
	options: RepeatedSuffixOptions = {}
): RepeatedSuffix | null {
	const minPeriodChars = options.minPeriodChars ?? 96;
	const maxPeriodChars = options.maxPeriodChars ?? 2048;
	const repetitions = options.repetitions ?? 3;
	const maxWindowChars = options.maxWindowChars ?? 8192;

	if (
		minPeriodChars < 1 ||
		maxPeriodChars < minPeriodChars ||
		repetitions < 2 ||
		maxWindowChars < minPeriodChars * repetitions
	) {
		return null;
	}

	const suffix = text.slice(-maxWindowChars);
	const maximum = Math.min(maxPeriodChars, Math.floor(suffix.length / repetitions));

	for (let period = minPeriodChars; period <= maximum; period++) {
		const finalStart = suffix.length - period;
		const finalBlock = suffix.slice(finalStart);

		let matches = true;

		for (let repetition = 2; repetition <= repetitions; repetition++) {
			const start = suffix.length - repetition * period;

			if (suffix.slice(start, start + period) !== finalBlock) {
				matches = false;

				break;
			}
		}

		if (matches) {
			// A long window made entirely from a tiny cycle ("no no no ...") is
			// ordinary token/word repetition and belongs to the existing samplers,
			// not this long-form reasoning-loop guard.
			let hasShortPrimitivePeriod = false;

			for (let shortPeriod = 1; shortPeriod < minPeriodChars; shortPeriod++) {
				if (period % shortPeriod !== 0) continue;

				let shortMatches = true;

				for (let index = shortPeriod; index < period; index++) {
					if (finalBlock[index] !== finalBlock[index % shortPeriod]) {
						shortMatches = false;

						break;
					}
				}

				if (shortMatches) {
					hasShortPrimitivePeriod = true;

					break;
				}
			}

			if (hasShortPrimitivePeriod) continue;

			return {
				periodChars: period,
				repeatedChars: period * repetitions,
				repetitions
			};
		}
	}

	return null;
}
