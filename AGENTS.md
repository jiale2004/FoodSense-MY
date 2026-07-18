# Repository Working Rules

These instructions apply to the entire FoodSense-MY repository.

## Documentation Is Part of Every Change

Whenever an implementation, bug fix, refactor, data-pipeline change, model-workflow change, configuration change, or user-facing behavior change is made, update the relevant documentation in the same change.

At minimum, review and update these files when applicable:

- `docs/handoff.md` for current progress, completed work, generated-data counts, operational status, and next steps.
- `docs/architecture.md` for system structure, module responsibilities, data flow, interfaces, scripts, and lifecycle changes.
- `docs/bounding-box-policy.md` for annotation, rejection, class interpretation, or quality-control policy changes.
- `README.md` for setup, commands, configuration, and user-facing usage changes.

Documentation must describe the implemented repository state, not planned behavior. Reconcile reported dataset counts and identifiers with generated manifests, summaries, or reports before recording them.

Before finishing a change:

1. Identify which documentation is affected.
2. Update it alongside the code or data-pipeline fix.
3. Verify commands, paths, counts, class IDs, and status statements against the repository.
4. Run `git diff --check` and the relevant tests.
5. Mention the documentation updates in the final handoff.

If a change truly has no documentation impact, state that explicitly in the final handoff. Do not silently skip the documentation review.
