# Simplifying the DCLab interface

**19 November 2026 · Design brief awaiting screenshots**

The owner wants a quieter workspace with fewer tabs competing for attention. People should still be able to reach the tools they use today.

## Screenshot review

For each supplied screen, identify the user's main task, the action that advances it and the tabs or controls competing with it. Trace each candidate to its existing destination. Record what to keep visible, what can move under an existing group, and what could be temporarily hidden. Use the screenshots to choose the first screen and document the reason for each proposed change.

## Implementation handoff

Prepare an English plan from that review. For each affected screen, include the current layout, proposed layout, route or component, how to reach each moved control, and a focused acceptance check. Use reversible presentation changes. Preserve existing functionality and source. If a control must leave the visible layout, retain its implementation and document how to restore it.

## Acceptance

- The main task and next action are visible on the reviewed screen.
- The revised default view has fewer competing tabs or controls, with before-and-after captures at the same viewport.
- Every moved action still works from its documented location. A temporarily hidden action can be restored without reconstructing its implementation.
- Existing deep links still resolve, and keyboard navigation follows the visible controls.

Specific screens, controls and before-and-after comparisons will be added after the screenshots arrive.
