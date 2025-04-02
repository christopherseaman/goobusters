# MD.ai Annotation Workflow for Optical Flow Tracking

## Current Problem

- MD.ai API doesn't support deleting annotations
- Running the script multiple times creates duplicate annotations
- Need a clean way to handle this limitation

## Proposed Workflow

-->Before Running the Script

- Create a new label in the MD.ai interface (e.g., "Fluid-OF-20250402")
- Note the generated label ID (e.g., "L_abc123")
- Optionally create a new label group if organizing by batch

-->Script Configuration

Add command-line options to specify the label ID: 

```bash
   python script.py --upload --label-id L_abc123
```

-->Running the Script

- When the --upload flag is present, annotations will be uploaded to MD.ai. 
- Each mask will be uploaded with the specified label ID. 
- All masks for the current run will use the same label ID. 

-->Managing Annotations

- Each script run uses a different label ID (manually created beforehand directly on the interface).
- Old annotations can be removed by deleting the entire label in MD.ai (this can also be done in bulk).
- This approach keeps annotations organised by processing batch.

-->What to include

- I'll create documentation explaining this in detail in this md. 
- The script will also warn users when uploading and remind them of this process.
- Each output run will include metadata about which label ID was used.

