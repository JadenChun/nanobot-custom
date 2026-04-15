# Frame Selection

Use this reference when a task depends on understanding a video through still frames.

## Goal

Create a storyboard that preserves the moments a human editor would care about:
- when the visible state changes
- when loading starts or ends
- when text becomes readable
- when proof, payoff, warning, or CTA screens appear

When OCR is available, include the extracted text in the storyboard so a smaller model can reason about the video without re-reading every image first.

When semantic summarization is available, prefer a separate semantic storyboard that states:
- what screen or state is visible
- what action or transition is happening
- why the moment matters to the viewer

## What to keep

Always prefer frames that explain the video's logic:
- the first meaningful state
- submit or transition moments
- the first readable result state
- warnings, badges, confirmations, or summaries
- the final state used for CTA or conclusion

## What to compress

Quiet stretches still need checkpoints, but not wall-to-wall frames:
- waiting after the viewer already understands what is happening
- repeated loading or scrolling with no new information
- static tails after the point has been made

## Screen-recording bias

For screen recordings, visual importance is usually tied to state changes rather than cinematic cuts.

Look for:
- new screens
- changed form values
- typed text appearing
- button clicks that trigger a new state
- spinners disappearing
- result panels, summaries, or badges becoming visible

## Limits

Still frames do not tell you everything:
- subtle motion can be lost
- timing rhythm still needs the timestamps
- audio intent still needs script, subtitles, or voiceover context

Use the storyboard as visual evidence, not as a substitute for every other source of truth.
