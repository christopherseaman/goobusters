# MD.ai Annotation Workflow for Optical Flow Tracking
Current Problem

MD.ai API doesn't support deleting annotations
Running the script multiple times creates duplicate annotations
Need a clean way to handle this limitation

Proposed Workflow
Before Running the Script

Create a new label in the MD.ai interface (e.g., "Fluid-OF-20250402")
Note the generated label ID (e.g., "L_abc123")
Optionally create a new label group if organizing by batch

Script Configuration

Add command-line options to specify the label ID: 

'python script.py --upload --label-id L_abc123'

