#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path so we can import project modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.ground_truth.create_ground_truth import create_ground_truth_for_exam

# These are examples - you'll need to replace with your actual issue type exams
ISSUE_TYPE_EXAMS = [
    {
        "name": "Exam 68",
        "study_uid": "1.2.826.0.1.3680043.8.498.89831828076854247086013149207529572702",
        "series_uid": "1.2.826.0.1.3680043.8.498.10865369767179070053130180937743816419",
        "project_id": "x9N2LJBZ",
        "dataset_id": "D_V688LQ",
        "label_id": "L_7DRjNJ"
    },
    # Add more issue type exams here
]

def main():
    parser = argparse.ArgumentParser(description="Create ground truth annotations for multiple exams")
    parser.add_argument("--output-dir", default="src/output/ground_truth", 
                      help="Directory to save ground truth annotations")
    parser.add_argument("--only-exam", default=None,
                      help="Only process a specific exam by name (e.g., 'Exam 68')")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process selected exams
    for exam in ISSUE_TYPE_EXAMS:
        if args.only_exam and args.only_exam != exam["name"]:
            continue
            
        print(f"\n===== Processing {exam['name']} =====")
        
        # Create exam-specific output directory
        exam_dir = os.path.join(args.output_dir, exam["name"].replace(" ", "_").lower())
        os.makedirs(exam_dir, exist_ok=True)
        
        try:
            create_ground_truth_for_exam(
                study_uid=exam["study_uid"],
                series_uid=exam["series_uid"],
                project_id=exam["project_id"],
                dataset_id=exam["dataset_id"],
                label_id=exam["label_id"],
                output_dir=exam_dir
            )
            print(f"✅ Ground truth created for {exam['name']}")
            
        except Exception as e:
            print(f"❌ Error creating ground truth for {exam['name']}: {str(e)}")
    
    print("\nGround truth creation process complete")

if __name__ == "__main__":
    main() 