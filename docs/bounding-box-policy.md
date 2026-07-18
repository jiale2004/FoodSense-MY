# FoodSense-MY Bounding-Box Policy

**Version:** 1.1
**Effective date:** 18 July 2026  
**Scope:** CVAT and YOLO annotations for the six FoodSense-MY target classes.

## Objective

Create consistent object-detection ground truth for Nasi Lemak, Roti Canai, Char Kuey Teow, Chicken Rice, Laksa, and Mee Goreng. The annotation must describe what is visibly present, not what the image filename, folder, search query, or source dataset claims.

## Core Rules

1. Annotate every confidently identifiable instance of any target class in the image.
2. Use one tight axis-aligned rectangle per distinct serving.
3. Include the complete edible serving and its integral components.
4. Exclude unrelated table space, hands, people, menus, packaging, and other dishes where practical.
5. Do not create a box for a non-target dish.
6. Do not force the source class when the visible food belongs to another target class.
7. If none of the six target classes is present, leave the CVAT frame without boxes and record it for rejection during import.

## What the Box Includes

The box should contain the dish as served, including components that define the serving:

- Nasi Lemak: coconut rice and the accompanying components presented as one plate or packet, such as sambal, egg, anchovies, peanuts, or cucumber.
- Chicken Rice: the chicken and rice serving, including closely arranged standard accompaniments on the same plate or tray.
- Roti Canai: the roti serving itself; include curry only when it is physically part of the same plated serving and a single tight box remains meaningful.
- Laksa: the bowl, plate, or container region containing the noodles, gravy or broth, and visible toppings; avoid excessive surrounding table. The class includes visually diverse Malaysian variants such as curry laksa and the less soupy Laksa Johor, so broth colour, noodle type, or unusual seafood toppings alone are not rejection reasons.
- Char Kuey Teow and Mee Goreng: the complete noodle portion and visible integral toppings on the plate or in the container.

For a bowl or plate whose food reaches its edges, the box may closely follow the vessel because the food boundary is otherwise ambiguous. Do not routinely box the entire table setting.

## Multiple Objects and Classes

- Use separate boxes for clearly separate servings, even if they share a class.
- When two target dishes appear in one image, annotate both with their actual classes.
- A single mixed plate that is conventionally one target dish gets one box for that dish.
- Do not create overlapping duplicate boxes for the same serving.

## Cropping, Occlusion, and Scale

- Annotate a partially cropped serving if its target class is still confidently identifiable; extend the box only to the visible image boundary.
- Annotate an occluded serving if its visible extent is sufficient for confident classification; box the estimated full visible serving boundary without inventing large invisible regions.
- Skip tiny, blurred, or heavily occluded dishes when the target class cannot be determined reliably.
- Background posters, screen images, logos, and illustrations are not real food instances and should not be annotated.

## Ambiguous Classes

Char Kuey Teow and Mee Goreng require special care. Use visible preparation cues rather than the source folder:

- Char Kuey Teow commonly uses flat rice noodles and a darker stir-fried appearance.
- Mee Goreng commonly uses yellow wheat noodles and its characteristic fried-noodle presentation.
- If the evidence is insufficient, flag the frame for review instead of guessing.

The audited pilot demonstrated that source labels are unreliable for this pair. Its final reviewed assignments include 33 source Mee Goreng images corrected to Char Kuey Teow and 7 in the reverse direction. One initial reverse correction was restored to Char Kuey Teow during Phase A.

Laksa also requires regional awareness. Curry laksa may contain tofu puffs, fish cake, prawns, mixed noodles, or rich coconut broth. Laksa Johor may appear almost dry, use spaghetti-like noodles, and be garnished with cucumber, onion, herbs, and lime. When the visual evidence is plausible but the variant is unfamiliar, flag it for review instead of rejecting it as generic noodles.

## Rejection Policy

An image should be rejected from the usable six-class staging tree when:

- none of the target classes is present;
- the food is not identifiable enough to assign a target class;
- only a menu, drawing, logo, or non-food representation is visible;
- corruption or severe quality issues prevent useful annotation.

Rejected images are evidence, not disposable files. The import process moves them to `data/dataset3/rejected/<batch>/`, records them in the manifest, and retains batch-level rejection metadata.

## CVAT Workflow

1. Confirm the six CVAT labels use the fixed canonical order.
2. Inspect the whole frame before drawing the first box.
3. Draw all target instances, including classes different from the source selection.
4. Zoom in to make box edges tight and check for duplicate boxes.
5. Flag uncertain class decisions for review.
6. Leave a truly non-target frame empty; do not add a placeholder box.
7. Export in Ultralytics YOLO Detection format.
8. Run the repository importer without `--apply` first, inspect its report, then rerun with `--apply` only after validation succeeds. For corrections to an already-merged batch, supply a new `--revision-id` so the importer creates a recoverable revision backup.

## Quality-Control Checklist

For each review batch, sample images from all six classes and verify:

- every visible target serving has a box;
- no non-target object has a box;
- the class reflects visible content rather than source provenance;
- boxes are tight and not duplicates;
- cropped and occluded cases follow the same rule;
- multi-class frames retain all valid boxes;
- empty frames are intentional rejections;
- exported class IDs are 0 through 5 and coordinates remain normalized within `[0, 1]`.

Any systematic ambiguity discovered during annotation should update this policy before the next batch so later labels remain consistent.
