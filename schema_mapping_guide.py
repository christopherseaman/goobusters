#!/usr/bin/env python3
"""
Comprehensive schema mapping guide for MD.ai annotations JSON file.
Provides multiple strategies for navigating and mapping the key tree structure.
"""

import json
import sys
from typing import Any, Dict, List, Tuple, Optional

class SchemaMapper:
    """A comprehensive schema mapper for the MD.ai annotations JSON structure."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.schema_cache = {}
    
    def find_by_uid(self, uid: str, uid_type: str = "StudyInstanceUID") -> List[Dict[str, Any]]:
        """
        Find records by any DICOM UID.
        
        Args:
            uid: The UID to search for
            uid_type: Type of UID (StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID)
        
        Returns:
            List of matching records with their paths and context
        """
        results = []
        
        for dataset_idx, dataset in enumerate(self.data.get('datasets', [])):
            for study_idx, study in enumerate(dataset.get('studies', [])):
                # Check study level
                if uid_type == "StudyInstanceUID" and study.get('StudyInstanceUID') == uid:
                    results.append({
                        'path': f"datasets[{dataset_idx}].studies[{study_idx}]",
                        'type': 'study',
                        'data': study,
                        'dataset_context': {
                            'id': dataset.get('id'),
                            'name': dataset.get('name'),
                            'type': dataset.get('type')
                        }
                    })
                
                # Check series level
                for series_idx, series in enumerate(study.get('series', [])):
                    if uid_type == "SeriesInstanceUID" and series.get('SeriesInstanceUID') == uid:
                        results.append({
                            'path': f"datasets[{dataset_idx}].studies[{study_idx}].series[{series_idx}]",
                            'type': 'series',
                            'data': series,
                            'study_context': {
                                'StudyInstanceUID': study.get('StudyInstanceUID'),
                                'number': study.get('number')
                            },
                            'dataset_context': {
                                'id': dataset.get('id'),
                                'name': dataset.get('name'),
                                'type': dataset.get('type')
                            }
                        })
                    
                    # Check instance level
                    for instance_idx, instance in enumerate(series.get('instances', [])):
                        if uid_type == "SOPInstanceUID" and instance.get('SOPInstanceUID') == uid:
                            results.append({
                                'path': f"datasets[{dataset_idx}].studies[{study_idx}].series[{series_idx}].instances[{instance_idx}]",
                                'type': 'instance',
                                'data': instance,
                                'series_context': {
                                    'SeriesInstanceUID': series.get('SeriesInstanceUID'),
                                    'number': series.get('number')
                                },
                                'study_context': {
                                    'StudyInstanceUID': study.get('StudyInstanceUID'),
                                    'number': study.get('number')
                                },
                                'dataset_context': {
                                    'id': dataset.get('id'),
                                    'name': dataset.get('name'),
                                    'type': dataset.get('type')
                                }
                            })
        
        return results
    
    def get_schema_map(self) -> Dict[str, Any]:
        """Get a comprehensive schema mapping."""
        if 'schema_map' in self.schema_cache:
            return self.schema_cache['schema_map']
        
        schema_map = {
            'root_structure': {
                'keys': list(self.data.keys()),
                'description': 'Top-level project structure'
            },
            'datasets': {
                'count': len(self.data.get('datasets', [])),
                'structure': self._analyze_datasets_structure(),
                'description': 'Each dataset contains studies, which contain series, which contain instances'
            },
            'dicom_hierarchy': {
                'levels': [
                    'Project (root)',
                    'Datasets (collections of studies)',
                    'Studies (DICOM Study level)',
                    'Series (DICOM Series level)', 
                    'Instances (DICOM Instance level)'
                ],
                'path_patterns': {
                    'StudyInstanceUID': 'datasets[i].studies[j].StudyInstanceUID',
                    'SeriesInstanceUID': 'datasets[i].studies[j].series[k].SeriesInstanceUID',
                    'SOPInstanceUID': 'datasets[i].studies[j].series[k].instances[l].SOPInstanceUID'
                }
            },
            'navigation_strategies': self._get_navigation_strategies()
        }
        
        self.schema_cache['schema_map'] = schema_map
        return schema_map
    
    def _analyze_datasets_structure(self) -> List[Dict[str, Any]]:
        """Analyze the structure of all datasets."""
        datasets_info = []
        
        for i, dataset in enumerate(self.data.get('datasets', [])):
            dataset_info = {
                'index': i,
                'id': dataset.get('id'),
                'name': dataset.get('name'),
                'type': dataset.get('type'),
                'studies_count': len(dataset.get('studies', [])),
                'total_series': sum(len(study.get('series', [])) for study in dataset.get('studies', [])),
                'total_instances': sum(
                    len(series.get('instances', [])) 
                    for study in dataset.get('studies', []) 
                    for series in study.get('series', [])
                )
            }
            datasets_info.append(dataset_info)
        
        return datasets_info
    
    def _get_navigation_strategies(self) -> Dict[str, List[str]]:
        """Get different strategies for navigating the schema."""
        return {
            'by_dataset': [
                "1. Start with datasets[i] where i is the dataset index",
                "2. Each dataset contains studies, series, and instances",
                "3. Use dataset.id or dataset.name to identify the right dataset"
            ],
            'by_study_uid': [
                "1. Search for StudyInstanceUID across all datasets",
                "2. Use find_by_uid(uid, 'StudyInstanceUID') method",
                "3. Get full context including dataset and study information"
            ],
            'by_series_uid': [
                "1. Search for SeriesInstanceUID across all datasets",
                "2. Use find_by_uid(uid, 'SeriesInstanceUID') method", 
                "3. Get full context including dataset, study, and series information"
            ],
            'by_instance_uid': [
                "1. Search for SOPInstanceUID across all datasets",
                "2. Use find_by_uid(uid, 'SOPInstanceUID') method",
                "3. Get full context including dataset, study, series, and instance information"
            ],
            'hierarchical_traversal': [
                "1. Start at root level: data['datasets']",
                "2. For each dataset: data['datasets'][i]",
                "3. For each study: data['datasets'][i]['studies'][j]",
                "4. For each series: data['datasets'][i]['studies'][j]['series'][k]",
                "5. For each instance: data['datasets'][i]['studies'][j]['series'][k]['instances'][l]"
            ]
        }
    
    def create_path_helper(self, target_uid: str) -> Dict[str, Any]:
        """Create a helper for finding a specific UID and its context."""
        # Try different UID types
        for uid_type in ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']:
            results = self.find_by_uid(target_uid, uid_type)
            if results:
                return {
                    'uid_type': uid_type,
                    'results': results,
                    'path_examples': [r['path'] for r in results],
                    'context_examples': [r for r in results]
                }
        
        return {'error': f'UID {target_uid} not found in any DICOM field'}

def main():
    json_file = "/Users/christopher/Documents/goobusters/data/mdai_ucsf_project_x9N2LJBZ_annotations_2025-09-18-050340.json"
    target_uid = ".4320960"
    
    print("Loading JSON file...")
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print("✓ JSON loaded successfully")
        
        # Create schema mapper
        mapper = SchemaMapper(data)
        
        # Find the target record
        print(f"\n" + "="*80)
        print(f"SEARCHING FOR RECORD CONTAINING '{target_uid}'")
        print("="*80)
        
        # Search for the specific UID that contains .4320960
        target_full_uid = "1.2.826.0.1.3680043.8.498.43209604241753848491338897571515450984"
        
        # Try different UID types
        for uid_type in ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']:
            results = mapper.find_by_uid(target_full_uid, uid_type)
            if results:
                print(f"✓ Found in {uid_type}:")
                for result in results:
                    print(f"  Path: {result['path']}")
                    print(f"  Type: {result['type']}")
                    if 'dataset_context' in result:
                        print(f"  Dataset: {result['dataset_context']['name']} (ID: {result['dataset_context']['id']})")
                    if 'study_context' in result:
                        print(f"  Study: {result['study_context']['StudyInstanceUID']}")
                    if 'series_context' in result:
                        print(f"  Series: {result['series_context']['SeriesInstanceUID']}")
                break
        else:
            print(f"✗ UID {target_full_uid} not found in any DICOM field")
        
        # Show comprehensive schema mapping
        print(f"\n" + "="*80)
        print("COMPREHENSIVE SCHEMA MAPPING")
        print("="*80)
        
        schema_map = mapper.get_schema_map()
        
        print(f"Root Structure: {schema_map['root_structure']['keys']}")
        print(f"Total Datasets: {schema_map['datasets']['count']}")
        
        print(f"\nDICOM Hierarchy:")
        for level in schema_map['dicom_hierarchy']['levels']:
            print(f"  - {level}")
        
        print(f"\nPath Patterns:")
        for uid_type, pattern in schema_map['dicom_hierarchy']['path_patterns'].items():
            print(f"  {uid_type}: {pattern}")
        
        print(f"\nNavigation Strategies:")
        for strategy_name, steps in schema_map['navigation_strategies'].items():
            print(f"\n{strategy_name.upper().replace('_', ' ')}:")
            for step in steps:
                print(f"  {step}")
        
        # Show dataset summary
        print(f"\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        
        for dataset_info in schema_map['datasets']['structure']:
            print(f"Dataset {dataset_info['index']}: {dataset_info['name']}")
            print(f"  ID: {dataset_info['id']}")
            print(f"  Type: {dataset_info['type']}")
            print(f"  Studies: {dataset_info['studies_count']}")
            print(f"  Series: {dataset_info['total_series']}")
            print(f"  Instances: {dataset_info['total_instances']}")
            print()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

