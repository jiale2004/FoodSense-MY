# CVAT Collaborator Guide

Use this guide when a FoodSense-MY group member is asked to review a prepared
image batch in CVAT. The repository owner prepares the ZIP files and performs
the final validated import into Dataset3. Reviewers should not modify Dataset3
directly.

## 1. What the reviewer receives

Each batch directory normally contains:

| File | Purpose |
|------|---------|
| `images.zip` | Images used to create the CVAT task |
| `preannotations.zip` | Model proposals for an assisted-training batch |
| `current-annotations.zip` | Existing human boxes for a holdout/audit batch |
| `selection.jsonl` | Repository provenance; do not upload or edit |
| `summary.json` | Expected image and proposal/box counts |

Upload exactly one annotation archive:

- assisted-training batch: `preannotations.zip`;
- holdout or audit batch: `current-annotations.zip`;
- fully manual batch with no starting boxes: do not upload an annotation ZIP.

Never import assisted-model proposals into a test-holdout task.

## 2. Fixed class mapping

The project must use these class IDs and names without reordering:

| ID | CVAT label |
|---:|------------|
| 0 | `nasi_lemak` |
| 1 | `roti_canai` |
| 2 | `char_kuey_teow` |
| 3 | `chicken_rice` |
| 4 | `laksa` |
| 5 | `mee_goreng` |

Use the existing CVAT project `FoodSense-MY dataset3` whenever possible so its
labels are inherited automatically. Do not create alternate spellings such as
`Chicken Rice` or `char kway teow`.

## 3. Create a task and upload `images.zip`

If the repository owner already created the task, skip to section 5 and open
the supplied job link.

1. Sign in to CVAT Online.
2. Open project `FoodSense-MY dataset3`.
3. Select **+** → **Create a new task**.
4. Enter the supplied task name, for example `dataset3 assisted batch 003`.
5. Confirm **Project** is `FoodSense-MY dataset3` and that project labels will
   be used.
6. Under **My computer**, upload the supplied `images.zip`.
7. Select **Submit & Open**.
8. Wait for media extraction to finish and verify the task's frame count against
   `summary.json`.

Do not upload multiple batches into one task. One repository batch must map to
one CVAT task and one local batch directory.

## 4. Import the starting annotations

1. On the task page, select **Actions** → **Upload annotations**.
2. For **Import format**, select exactly
   **Ultralytics YOLO Detection 1.0**.
3. Leave **Import mode** as **Replace** only when this is a newly created empty
   task. Never replace annotations in a task that somebody has already edited.
4. Upload the supplied `preannotations.zip` or `current-annotations.zip`.
5. Select **OK**, then confirm **Replace annotations** if CVAT asks.
6. Wait until the request is finished. The **Requests** page shows import
   progress if the task page closes the dialog.
7. Open the job, select **Info**, and compare the frame and rectangle totals
   with `summary.json` before reviewing.

Stop and contact the repository owner if:

- CVAT reports an unknown label or class mapping error;
- the frame count differs from `summary.json`;
- the import has zero rectangles when the summary expects proposals/boxes;
- the selected format is anything other than Ultralytics YOLO Detection 1.0.

## 5. Review every frame

The imported rectangles are starting points, not automatically accepted truth.

For every frame from `0` through the final frame:

1. Inspect the entire image, not only the existing rectangle.
2. Tighten loose boxes and expand boxes that cut off part of the serving.
3. Correct the class based on visible food, not the source filename or folder.
4. Delete boxes on non-target food or background objects.
5. Add every missed instance belonging to the six target classes.
6. Remove duplicate rectangles around the same serving.
7. Keep all valid target classes when several dishes appear in one image.
8. Save regularly with **Save** or `Ctrl/Cmd+S`.

If an image contains none of the six target classes, delete every rectangle and
leave the frame empty. An empty reviewed frame means **reject/quarantine this
image**; it is not a background training example.

For an assisted-training batch, treat every model rectangle as a proposal.
For a holdout task, do not run automatic annotation or consult model
predictions. Follow [bounding-box-policy.md](bounding-box-policy.md) for edge,
occlusion, crop, and class-ambiguity rules.

## 6. Finish the job

1. Confirm that the frame selector has reached the last frame.
2. Select **Info** and note the final per-class rectangle totals.
3. Select **Menu** → **Finish the job** → **Continue**.
4. Confirm the job state is **completed** on the task page.

Do not delete the task yet. The repository owner must first download, validate,
apply, and archive the reviewed export.

## 7. Export the reviewed annotations

1. Open the completed job.
2. Select **Menu** → **Export job dataset**. The task-level
   **Actions** → **Export task dataset** option is also acceptable for a
   one-job task.
3. Select **Ultralytics YOLO Detection 1.0**.
4. Leave **Save images** turned off. The repository already has the images.
5. Set **Custom name** to `cvat-reviewed-export`.
6. Select **OK**.
7. If no download starts immediately, open **Requests**, find the finished
   export for the correct task/job, select **…** → **Download**.
8. Do not unzip or edit the downloaded archive.

The final filename handed to the repository owner must be
`cvat-reviewed-export.zip`. Confirm it belongs to the correct task before
sending it.

## 8. Reviewer handoff checklist

Send the repository owner all of the following:

- task name and task ID;
- job ID and job link;
- confirmation that every frame was reviewed;
- final job state: `completed`;
- final rectangle total and per-class totals from **Info**;
- number or frame IDs of intentional empty/rejected images;
- any uncertain frames or policy questions;
- the untouched `cvat-reviewed-export.zip`.

Example:

```text
Task: dataset3 assisted batch 003 (#123456)
Job: #654321, completed
Reviewed: all frames 0-299
Final boxes: 315
Intentional empty frames: 8, 41, 177
Uncertain frames: 52 (noodle class), 203 (heavy occlusion)
Export: cvat-reviewed-export.zip
```

## 9. Repository-owner validation

Only the repository owner performs this stage. The first importer run is a dry
run and must omit `--apply`:

```bash
python training_scripts/import_cvat_annotations.py \
  --dataset-dir data/dataset3 \
  --pilot-dir data/cvat/<batch-directory> \
  --archive data/cvat/<batch-directory>/cvat-reviewed-export.zip \
  --batch-id <stable-batch-id> \
  --task-id <task-id> \
  --job-id <job-id>
```

For a correction or holdout audit of already annotated images, use a unique
`--revision-id` instead of `--batch-id`. The owner reviews the dry-run report,
then repeats the same command with `--apply`. The importer creates recovery
metadata before changing Dataset3.

## 10. Task-slot rotation

The free CVAT Online account supports only three simultaneous tasks. A completed
task may be deleted only after all of these are true:

- the reviewed export is stored locally;
- the importer dry-run passed;
- the apply completed;
- the revision/merge report and recovery files exist;
- repository counts and tests passed.

After those checks, delete the hosted task and reuse its slot for the next
batch. Local batch directories are the durable record; CVAT tasks are temporary
review workspaces.
